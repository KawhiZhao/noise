# Copyright 2021 Zhongyang Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import torch
import importlib
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs

import pytorch_lightning as pl
import pdb


class MInterfaceSTFT(pl.LightningModule):
    def __init__(self, model_name, loss, lr, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()
        self.min_angle_error = 100

    def forward(self, stft):
        return self.model(stft)

    def training_step(self, batch, batch_idx):
        stft, doa = batch
        # convert to float
        stft = stft.float()
        
        # caption_embedding = caption_embedding.float()
        estimated_azimuth = self(stft)
       
        loss = self.loss_function(estimated_azimuth, doa, estimated_azimuth.device)
      
        
        azimuth_angles = torch.argmax(estimated_azimuth, dim=-1)
        azimuth_error = torch.abs(azimuth_angles - doa)
        azimuth_error = torch.where(azimuth_error > 180, 360 - azimuth_error, azimuth_error)
        # calculate the accuracy of the azimuth with error less than 25 degree
        accuracy = torch.where(azimuth_error < 10, 1, 0)
        
        accuracy = torch.sum(accuracy) / accuracy.shape[0]
        azimuth_error = azimuth_error.float().mean()
        
        # pdb.set_trace()
        # log the angle error of azimuth and elevation
        self.log('train_azimuth_error', azimuth_error, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        # self.log('elevation_error', elevation_error, on_step=False, on_epoch=True, prog_bar=True)
        # log the overall loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        # return (azimuth_error, accuracy, loss)
        return loss

    def validation_step(self, batch, batch_idx):
        stft, doa = batch
        # convert to float
        stft = stft.float()
        
        # caption_embedding = caption_embedding.float()
        estimated_azimuth = self(stft)
       
        loss = self.loss_function(estimated_azimuth, doa, estimated_azimuth.device)
      
        
        azimuth_angles = torch.argmax(estimated_azimuth, dim=-1)
        azimuth_error = torch.abs(azimuth_angles - doa)
        azimuth_error = torch.where(azimuth_error > 180, 360 - azimuth_error, azimuth_error)
        # calculate the accuracy of the azimuth with error less than 25 degree
        accuracy = torch.where(azimuth_error < 5, 1, 0)
        # pdb.set_trace()
        accuracy = torch.sum(accuracy) / accuracy.shape[0]
        azimuth_error = azimuth_error.float().mean()
        
        
        # log the angle error of azimuth and elevation
        self.log('val_azimuth_error', azimuth_error, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('elevation_error', elevation_error, on_step=False, on_epoch=True, prog_bar=True)
        # log the overall loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return (azimuth_error, accuracy, loss)

    def test_step(self, batch, batch_idx):
        stft, doa = batch
        # convert to float
        stft = stft.float()
        
        # caption_embedding = caption_embedding.float()
        estimated_azimuth = self(stft)
       
        loss = self.loss_function(estimated_azimuth, doa, estimated_azimuth.device)
      
        
        azimuth_angles = torch.argmax(estimated_azimuth, dim=-1)
        azimuth_error = torch.abs(azimuth_angles - doa)
        azimuth_error = torch.where(azimuth_error > 180, 360 - azimuth_error, azimuth_error)
        # calculate the accuracy of the azimuth with error less than 25 degree
        accuracy = torch.where(azimuth_error < 5, 1, 0)
        # pdb.set_trace()
        accuracy = torch.sum(accuracy) / accuracy.shape[0]
        azimuth_error = azimuth_error.float().mean()
        
        
        # log the angle error of azimuth and elevation
        self.log('test_azimuth_error', azimuth_error, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('elevation_error', elevation_error, on_step=False, on_epoch=True, prog_bar=True)
        # log the overall loss
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return (azimuth_error, accuracy, loss)

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs)
    
    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs)
    
    def shared_epoch_end(self, outputs):
     
        azimuth_error = torch.stack([x[0] for x in outputs]).mean()
        accuracy = torch.stack([x[1] for x in outputs]).mean()
        loss = torch.stack([x[2] for x in outputs]).mean()
        self.log('val_azimuth_error_avg', azimuth_error, on_epoch=True, sync_dist=True)
        self.log('val_accuracy_avg', accuracy, on_epoch=True, sync_dist=True)
        self.log('val_loss_avg', loss, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def configure_loss(self):
        # self.angle_loss = EMD_loss(5)
        loss = self.hparams.loss.lower()
        if loss == 'mse':
            self.loss_function = F.mse_loss
        elif loss == 'l1':
            self.loss_function = F.l1_loss
        elif loss == 'bce':
            self.loss_function = F.binary_cross_entropy
        elif loss == 'ce':
            self.loss_function = F.cross_entropy
        elif loss == 'emd':
            self.loss_function = EMD_loss(5)
        else:
            raise ValueError("Invalid Loss Type!")

    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)
import torch.nn as nn
import numpy as np
import math
class EMD_loss(nn.Module):
    def __init__(self, sigma) -> None:
        super().__init__()
        self.sigma = sigma
    
    def forward(self, y_pred, y_labels, device):

        x = np.arange(0, 360, 1)
        batch_size = y_labels.shape[0]
        y_labels = np.array(y_labels.cpu())
        y_dist = np.ones((batch_size, 360))
        for i in range(batch_size):
            y_dist[i] = np.exp(-(x - y_labels[i]) ** 2 /(2* self.sigma **2))/(math.sqrt(2*math.pi)*self.sigma)
        y_dist = torch.Tensor(y_dist).to(device)

        return torch.mean(torch.sum(torch.square(y_dist - y_pred), dim=1), dim=0)
