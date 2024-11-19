# Copyright 2021 Zhongyang Zhang
# Contact: mirakuruyoo@gmai.com
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

""" This main entrance of the whole project.

    Most of the code should not be changed, please directly
    add all the input arguments of your model's constructor
    and the dataset file's constructor. The MInterface and 
    DInterface can be seen as transparent to all your args.    
"""
import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger, NeptuneLogger

from model import *
from data import DInterface
from utils import load_model_path_by_args
import torch
import pdb

def load_callbacks():
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='val_azimuth_error',
        mode='min',
        patience=10,
        min_delta=0.001
    ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='val_azimuth_error',
        filename='best-{epoch:02d}-{val_loss_end:.3f}',
        save_top_k=3,
        mode='min',
        save_last=True
    ))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))
    return callbacks


def main(args):
    # torch.set_float32_matmul_precision('medium')
    # torch.multiprocessing.set_start_method('spawn')# good solution !!!!

    pl.seed_everything(args.seed)
    # load_path = load_model_path_by_args(args)
    data_module = DInterface(**vars(args))
    load_path = args.load_path
    # if load_path is None:
    #     model = MInterface(**vars(args))
    # else:
    #     model = MInterface(**vars(args))
    #     args.ckpt_path = load_path
    # model = MInterfaceSeparateLoc(**vars(args))
    # model = MInterfaceV2(**vars(args))
    model = MInterfaceSTFT(**vars(args))
    print(model)
    # # If you want to change the logger's saving folder
    # logger = TensorBoardLogger(save_dir='kfold_log', name=args.log_dir)
    args.callbacks = load_callbacks()
    # Set the WANDB_CACHE_DIR environment variable
    # os.environ['WANDB_CACHE_DIR'] = '/mnt/fast/nobackup/scratch4weeks/jz01019/AAC-SELD/cache'

    # wandb_logger = WandbLogger(
    #     project='SELD', 
    #     name='Text-query-simulated-EMD',
    #     log_model='all',
    #     save_dir='save_files',
    #     # mode='disabled',
    #     )
    # args.logger = wandb_logger

    # logger = NeptuneLogger(
    #     api_key='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjOTFlODBmMC0zZGNkLTRjNjYtYmM2NC1lMzk0MTJiZDNkZDAifQ==',  
    #     project="zhaojinzheng98/Common",  
    #     tags=["training"],  # optional
    #     name="SELD Base xy prediction",  # optional
    # )   
    # wandb_logger = WandbLogger(
    #     project='noise adaptation', 
    #     # name='two channels 1, 2 -> 3', 
    #     name='snr10 stft resnet conformer simualted', 
    #     # mode='disabled',
    # )
    # args.logger = wandb_logger

    if args.load_path != '':
        trainer = Trainer.from_argparse_args(args, resume_from_checkpoint=load_path)
    else:
        trainer = Trainer.from_argparse_args(args)
    # pdb.set_trace()
    # trainer.fit(model, data_module)

    checkpoint = torch.load(args.load_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    # trainer.test(model, data_module)
    if not args.ddp:
        trainer.test(model, data_module)
    else:
        trainer = Trainer(devices=1, num_nodes=1)
        trainer.test(model, data_module)


if __name__ == '__main__':
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)

    # LR Scheduler
    parser.add_argument('--lr_scheduler', choices=['step', 'cosine'], type=str)
    parser.add_argument('--lr_decay_steps', default=20, type=int)
    parser.add_argument('--lr_decay_rate', default=0.5, type=float)
    parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)


    # Training Info
    parser.add_argument('--dataset', default='simulated_speech_stft', type=str)
    parser.add_argument('--data_dir', default='ref/data', type=str)
    parser.add_argument('--model_name', default='resnet_conformer', type=str)
    parser.add_argument('--loss', default='emd', type=str)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--log_dir', default='lightning_logs', type=str)
    # parser.add_argument('--load_path', default='/mnt/fast/nobackup/scratch4weeks/jz01019/AAC-SELD/checkpoints/best-epoch=27-val_loss_end=0.042.ckpt', type=str)
    parser.add_argument('--load_path', default='noise adaptation/fz79k5qo/checkpoints/best-epoch=37-val_loss_end=0.000.ckpt', type=str)
    # train from scratch
    # parser.add_argument('--load_path', default='', type=str)
    parser.add_argument('--snr', default=-1, type=int)
    parser.add_argument('--ddp', default=False, type=str)
    # Add pytorch lightning's args to parser as a group.
    parser = Trainer.add_argparse_args(parser)

    ## Deprecated, old version
    # parser = Trainer.add_argparse_args(
    #     parser.add_argument_group(title="pl.Trainer args"))

    # Reset Some Default Trainer Arguments' Default Values

    parser.set_defaults(
        max_epochs=100, 
        accelerator='gpu', 
        devices=1, 
        precision='32', 
        # val_check_interval = 10,
        # fast_dev_run=50,
        logger=False,
        limit_val_batches=1.0,
        check_val_every_n_epoch=2,
        # nb_sanity_val_steps=0,
        )

    args = parser.parse_args()

    main(args)
