import torch
import numpy as np
import sys
import os 
import torch.nn as nn
import torch.nn.functional as F
import yaml
import pickle
from model import AE, LatentDiscriminator
from data_utils import get_data_loader
from data_utils import PickleDataset
from utils import *
from functools import reduce


class Solver(object):
    def __init__(self, config, args):
        # config store the value of hyperparameters, turn to attr by AttrDict
        self.config = AttrDict(config)
        print(config)

        # args store other information
        self.args = args
        print(self.args)

        # logger to use tensorboard
        self.logger = Logger(self.args.logdir)

        # get dataloader
        self.get_data_loaders()

        # init the model with config
        self.build_model()

        if args.load_model:
            self.load_model(args.load_opt, args.load_dis)

    def save_model(self, iteration):
        # save model and discriminator and their optimizer
        torch.save(self.model.state_dict(), f'{self.args.store_model_path}-{iteration}.ckpt')
        torch.save(self.gen_opt.state_dict(), f'{self.args.store_model_path}-{iteration}.opt')
        torch.save(self.discr.state_dict(), f'{self.args.store_model_path}-{iteration}.discr')
        torch.save(self.dis_opt.state_dict(), f'{self.args.store_model_path}-{iteration}.discr.opt')

    def load_model(self, load_opt, load_dis):
        print(f'Load model from {args.load_model_path}')
        self.model.load_state_dict(torch.load(f'{args.load_model_path}.ckpt'))
        if load_dis:
            self.discr.load_state_dict(torch.load(f'{args.load_model_path}.discr'))
        if load_opt:
            self.gen_opt.load_state_dict(torch.load(f'{args.load_model_path}.opt'))
        if load_dis and load_opt:
            self.dis_opt.load_state_dict(torch.load(f'{args.load_model_path}.discr.opt'))
        return

    def get_data_loaders(self):
        data_dir = self.args.data_dir

        self.train_dataset = PickleDataset(os.path.join(data_dir, f'{self.args.train_set}.pkl'), 
                os.path.join(data_dir, self.args.train_index_file), 
                segment_size=self.config.segment_size)

        self.val_dataset = PickleDataset(os.path.join(data_dir, f'{self.args.val_set}.pkl'), 
                os.path.join(data_dir, self.args.val_index_file), 
                segment_size=self.config.segment_size)

        self.train_loader = get_data_loader(self.train_dataset, 
                batch_size=self.config.batch_size, 
                shuffle=self.config.shuffle, 
                num_workers=4, drop_last=False)

        self.val_loader = get_data_loader(self.val_dataset, 
                batch_size=self.config.batch_size, 
                shuffle=self.config.shuffle, 
                num_workers=4, drop_last=False)
        return

    def build_model(self): 
        # create model, discriminator, optimizers
        self.model = AE(c_in=self.config.c_in,
                c_h=self.config.c_h,
                c_out=self.config.c_in,
                c_cond=self.config.c_cond,
                kernel_size=self.config.kernel_size,
                bank_size=self.config.bank_size,
                bank_scale=self.config.bank_scale,
                s_enc_n_conv_blocks=self.config.s_enc_n_conv_blocks,
                s_enc_n_dense_blocks=self.config.s_enc_n_dense_blocks,
                d_enc_n_conv_blocks=self.config.d_enc_n_conv_blocks,
                d_enc_n_dense_blocks=self.config.d_enc_n_dense_blocks,
                s_subsample=self.config.s_subsample,
                d_subsample=self.config.d_subsample,
                dec_n_conv_blocks=self.config.dec_n_conv_blocks,
                dec_n_dense_blocks=self.config.dec_n_dense_blocks,
                upsample=self.config.upsample,
                act=self.config.act,
                dropout_rate=self.config.dropout_rate)
        print(self.model)

        discr_input_size = self.config.segment_size * reduce(lambda x, y: x*y, self.config.d_subsample)
        self.discr = LatentDiscriminator(input_size=discr_input_size,
                c_in=self.config.c_h * 2, 
                c_h=self.config.dis_c_h, 
                kernel_size=self.config.kernel_size,
                n_conv_layers=self.config.dis_n_conv_layers, 
                d_h=self.config.dis_d_h, 
                act=self.config.act, 
                dropout_rate=self.config.dis_dropout_rate)
        print(self.discr)
        self.gen_opt = torch.optim.Adam(self.model.parameters(), 
                lr=self.config.gen_lr, betas=(self.config.beta1, self.config.beta2))  
        self.dis_opt = torch.optim.Adam(self.discr.parameters(), 
                lr=self.config.dis_lr, betas=(self.config.beta1, self.config.beta2)) 
        print(self.gen_opt)
        print(self.dis_opt)
        return

if __name__ == '__main__':
    with open('./config.yaml') as f:
        config = yaml.load(f)
    solver = Solver(config, args)
