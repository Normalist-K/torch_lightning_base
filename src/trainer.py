import os
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import wandb
from warmup_scheduler import GradualWarmupScheduler

from src.models.components.model_ema import ModelEMA
from src.utils.metric import calc_metric


class Trainer:
    def __init__(self, cfg, model, device, verbose=True):
        self.cfg = cfg
        self.model = model.to(device)
        if cfg.trainer.model_ema:
            self.model_ema = ModelEMA(self.model)
        self.device = device
        self.verbose = verbose

        # Loss function
        if cfg.trainer.criterion == 'CrossEntropyLoss':
            criterion = nn.CrossEntropyLoss(reduction='none')
        elif cfg.trainer.criterion == 'MSELoss':
            criterion = nn.MSELoss()

        # Optimizer
        if cfg.trainer.optimizer.name.lower() == 'adamw':
            optimizer = optim.AdamW(self.model.parameters(),
                                    lr=cfg.trainer.optimizer.lr,
                                    weight_decay=cfg.trainer.optimizer.weight_decay)
        elif cfg.trainer.optimizer.name.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(),
                                   lr=cfg.trainer.optimizer.lr,
                                   weight_decay=cfg.trainer.optimizer.weight_decay)
        elif cfg.trainer.optimizer.name.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(),
                                  lr=cfg.trainer.optimizer.lr,
                                  weight_decay=cfg.trainer.optimizer.weight_decay,
                                  momentum=cfg.trainer.optimizer.momentum)

        # Scheduler
        if cfg.trainer.scheduler.name == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             'min',
                                                             factor=cfg.trainer.scheduler.lr_factor,
                                                             patience=cfg.trainer.scheduler.patience,
                                                             verbose=verbose)
        elif cfg.trainer.scheduler.name == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                             T_max=(cfg.len_train_loader * cfg.epoch),
                                                             eta_min=cfg.trainer.scheduler.min_lr)
        else:
            scheduler = None

        if scheduler is not None and cfg.trainer.scheduler.warmup:
            scheduler = GradualWarmupScheduler(optimizer, 
                                               multiplier=1, 
                                               total_epoch=3*cfg.len_train_loader, 
                                               after_scheduler=scheduler)

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.start_epoch = 0
        if cfg.resume:
            checkpoint = torch.load(cfg.path.pretrained)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            print(f"Model loaded: {cfg.path.pretrained}")
        
        self.best_model = deepcopy(self.model)
        self.best_valid_metric = np.Inf
        self.es_patience = 0
        self.scaler = GradScaler() if cfg.mixed_precision else None
        
    def fit(self, train_loader, valid_loader):
        
        # 1 epoch
        for epoch in range(self.start_epoch, self.start_epoch + self.cfg.epoch):
            
            # Train
            train_loss = self.train_epoch(train_loader)
            print(f'\nEpoch: {epoch}')
            print(f'Train Loss: {train_loss:.6f}')

            # Validation
            valid_loss, valid_metric = self.validation(valid_loader)
            print(f'Valid Loss: {valid_loss:.6f}, Valid metric: {valid_metric:.6f}')
            
            if self.cfg.trainer.scheduler.name == 'ReduceLROnPlateau':
                self.scheduler.step(valid_loss)
            
            # Log
            if not self.cfg.DEBUG:
                wandb.log({"train_loss": train_loss,
                           "valid_loss": valid_loss, 
                           "valid_metric": valid_metric,
                           "lr": self.optimizer.param_groups[0]['lr'],
                           })
        
            # Model EMA & Early stopping & Model save
            if valid_metric < self.best_valid_metric:
                if self.cfg.trainer.model_ema:
                    self.best_model = deepcopy(self.model_ema.ema)
                else:
                    self.best_model = deepcopy(self.model)
                self.best_valid_metric = valid_metric
                self.es_patience = 0
                if not self.cfg.DEBUG:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.best_model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'metric': valid_metric,
                        'loss': valid_loss,
                    }, os.path.join(self.cfg.path.weights, f'{self.cfg.name}-{self.cfg.dt_string}/{epoch}.pth')) 
                    print(f'Epoch {epoch} Model saved. ({self.cfg.name}-{self.cfg.dt_string}/{epoch}.pth)')
            elif epoch == (self.start_epoch + self.cfg.epoch - 1):
                if self.cfg.trainer.model_ema:
                    save_model = deepcopy(self.model_ema.ema)
                else:
                    save_model = deepcopy(self.model)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': save_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss': valid_loss,
                }, os.path.join(self.cfg.path.weights, f'{self.cfg.name}-{self.cfg.dt_string}/{epoch}_last.pth')) 
                print(f'Epoch {epoch} Model saved. ({self.cfg.name}-{self.cfg.dt_string}/{epoch}_last.pth)')
            else:
                self.es_patience += 1
                print(f"Valid metric. increased. Current early stop patience is {self.es_patience}")

            if (self.cfg.es_patience != 0) and (self.es_patience == self.cfg.es_patience):
                break

    def train_epoch(self, train_loader):
        self.model.train()

        losses = []
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=True, position=0, desc='Train')
        for batch_idx, (data_path, x, y) in pbar:

            self.optimizer.zero_grad()
            
            x = x.to(self.device)
            y = y.to(self.device)
        
            current_lr = self.optimizer.param_groups[0]['lr']

            if self.cfg.mixed_precision:
                with autocast():
                    pred = self.model(x)
                    loss = self.criterion(pred, y)

                self.scaler.scale(loss).backward()

                if self.cfg.trainer.optimizer.clipping:
                    # Gradient Norm Clipping
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0, norm_type=2)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                pred = self.model(x)
                loss = self.criterion(pred, y)

                loss.backward()

                if self.cfg.trainer.optimizer.clipping:
                    # Gradient Norm Clipping
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0, norm_type=2)
                
                self.optimizer.step()

            losses.append(loss.cpu().item())
            pbar.set_postfix(loss=loss.cpu().item())
            
            if self.cfg.trainer.model_ema:
                self.model_ema.update(self.model)

            if (self.scheduler is not None) and (self.cfg.trainer.scheduler.name != 'ReduceLROnPlateau'):
                self.scheduler.step()
            
            if self.cfg.DEBUG and batch_idx > 5:
                break

        return np.average(losses)

    def validation(self, valid_loader):
        if self.cfg.trainer.model_ema:
            model = self.model_ema.ema.eval()
        else:
            model = self.model.eval()

        losses, metrics = [], []
        p_bar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Valid', position=0, leave=True)
        for batch_idx, (data_path, x, y) in p_bar:
            
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.no_grad():
                pred = self.model(x)
                loss = self.criterion(pred, y)
                
                metric = calc_metric(pred, y)
                metrics.append(metric)
                
                if not self.cfg.DEBUG and batch_idx == 0:
                    # check results
                    pass


                losses.append(loss.cpu().item())

        torch.cuda.empty_cache()

        return np.average(losses), np.average(metrics)

    def inference(self, test_loader):
        self.best_model = self.best_model.to(self.device)
        self.best_model.eval()
        
        result_ids = []
        result_preds = []

        p_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Infer', position=0, leave=True)
        for i, (batch_id, x) in p_bar:
            torch.cuda.empty_cache()
            
            x = x.to(self.device)

            with torch.no_grad():
                batch_pred = self.best_model(x)

                for id, pred in zip(batch_id, batch_pred):
                    result_ids.append(id)
                    result_preds.append(pred)

        return result_ids, result_preds