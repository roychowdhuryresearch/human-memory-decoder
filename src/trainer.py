import torch
import torch.nn as nn
import numpy as np  
import os
from tqdm import tqdm
from src.utils.meters import *
from src.models.ensemble import Ensemble
from src.utils.initializer import *

class Trainer:
    def __init__(
        self,
        model: Ensemble,
        evaluator,
        optimizers,
        lr_scheduler,
        data_loaders,
        config,
    ):
        super().__init__()
        self.model = model
        self.evaluator = evaluator
        self.optimizer = optimizers
        self.lr_scheduler = lr_scheduler
        self.train_loader = data_loaders["train"]
        self.valid_loader = data_loaders["valid"]
        self.inference_loader = data_loaders['inference']
        self.device = config['device']
        self.config = config

        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def train(self, epochs, fold):
        for epoch in range(epochs):
            self.model.train()
            meter = Meter(fold)

            frame_index = np.empty(0)
            y_pred = np.empty((0, self.config['num_labels']))
            y_true = np.empty((0, self.config['num_labels']))
            
            # Create tqdm progress bar
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), 
                       desc=f'Epoch {epoch+1}/{epochs}')
            
            for i, (feature, target, index) in pbar:
                target = target.to(self.device)
                if not self.config['use_lfp'] and self.config['use_spike']:
                    spike = feature.to(self.device)
                    lfp = None
                elif self.config['use_lfp'] and not self.config['use_spike']:
                    lfp = feature.to(self.device)
                    spike = None
                else:
                    assert isinstance(feature, list) or isinstance(feature, tuple), "Tensor must be a list or tuple"
                    spike = feature[1].to(self.device)
                    lfp = feature[0].to(self.device)
                # forward pass
                spike_emb, lfp_emb, output, attentions = self.model(lfp, spike)
                bce_loss = self.bce_loss(output, target)
                bce_loss = torch.mean(bce_loss)
                loss = bce_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # prediction
                output = torch.sigmoid(output)
                pred = np.round(output.cpu().detach().numpy())
                true = np.round(target.cpu().detach().numpy())
                y_pred = np.concatenate([y_pred, pred], axis=0)
                y_true = np.concatenate([y_true, true], axis=0)
                accuracy = self.evaluator.calculate_accuracy(true, pred)
                f1 = self.evaluator.calculate_f1(true, pred)
                frame_index = np.concatenate([frame_index, index], axis=0)
                meter.add(loss.item(), f1, accuracy)
                
                # Update progress bar with current metrics
                metrics = meter.get_current_metrics()
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'f1': f"{metrics['f1']:.4f}",
                    'acc': f"{metrics['accuracy']:.4f}"
                })

            self.lr_scheduler.step()

            if (epoch+1) in self.config['save_epochs']:
                model_save_path = os.path.join(self.config['train_save_path'], 'model_weights_epoch{}.tar'.format(epoch+1))
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'args': self.config
                }, model_save_path)

                print()
