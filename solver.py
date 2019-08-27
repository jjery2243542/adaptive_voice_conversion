import torch
import numpy as np
import sys
import os 
import torch.nn as nn
import torch.nn.functional as F
import yaml
import pickle
from model import AE
from data_utils import get_data_loader
from data_utils import PickleDataset
from utils import *
from functools import reduce
from collections import defaultdict

class Solver(object):
    def __init__(self, config, args):
        # config store the value of hyperparameters, turn to attr by AttrDict
        self.config = config
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
        self.save_config()

        if args.load_model:
            self.load_model()

    def save_model(self, iteration):
        # save model and discriminator and their optimizer
        torch.save(self.model.state_dict(), f'{self.args.store_model_path}.ckpt')
        torch.save(self.opt.state_dict(), f'{self.args.store_model_path}.opt')

    def save_config(self):
        with open(f'{self.args.store_model_path}.config.yaml', 'w') as f:
            yaml.dump(self.config, f)
        with open(f'{self.args.store_model_path}.args.yaml', 'w') as f:
            yaml.dump(vars(self.args), f)
        return

    def load_model(self):
        print(f'Load model from {self.args.load_model_path}')
        self.model.load_state_dict(torch.load(f'{self.args.load_model_path}.ckpt'))
        self.opt.load_state_dict(torch.load(f'{self.args.load_model_path}.opt'))
        return

    def get_data_loaders(self):
        data_dir = self.args.data_dir
        self.train_dataset = PickleDataset(os.path.join(data_dir, f'{self.args.train_set}.pkl'), 
                os.path.join(data_dir, self.args.train_index_file), 
                segment_size=self.config['data_loader']['segment_size'])
        self.train_loader = get_data_loader(self.train_dataset,
                frame_size=self.config['data_loader']['frame_size'],
                batch_size=self.config['data_loader']['batch_size'], 
                shuffle=self.config['data_loader']['shuffle'], 
                num_workers=4, drop_last=False)
        self.train_iter = infinite_iter(self.train_loader)
        return

    def build_model(self): 
        # create model, discriminator, optimizers
        self.model = cc(AE(self.config))
        print(self.model)
        optimizer = self.config['optimizer']
        self.opt = torch.optim.Adam(self.model.parameters(), 
                lr=optimizer['lr'], betas=(optimizer['beta1'], optimizer['beta2']), 
                amsgrad=optimizer['amsgrad'], weight_decay=optimizer['weight_decay'])
        print(self.opt)
        return

    def ae_step(self, data, lambda_kl):
        x = cc(data)
        mu, log_sigma, emb, dec = self.model(x)
        criterion = nn.L1Loss()
        loss_rec = criterion(dec, x)
        loss_kl = 0.5 * torch.mean(torch.exp(log_sigma) + mu ** 2 - 1 - log_sigma)
        loss = self.config['lambda']['lambda_rec'] * loss_rec + \
                lambda_kl * loss_kl
        self.opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                max_norm=self.config['optimizer']['grad_norm'])
        self.opt.step()
        meta = {'loss_rec': loss_rec.item(),
                'loss_kl': loss_kl.item(),
                'grad_norm': grad_norm}
        return meta

    def train(self, n_iterations):
        for iteration in range(n_iterations):
            if iteration >= self.config['annealing_iters']:
                lambda_kl = self.config['lambda']['lambda_kl']
            else:
                lambda_kl = self.config['lambda']['lambda_kl'] * (iteration + 1) / self.config['annealing_iters'] 
            data = next(self.train_iter)
            meta = self.ae_step(data, lambda_kl)
            # add to logger
            if iteration % self.args.summary_steps == 0:
                self.logger.scalars_summary(f'{self.args.tag}/ae_train', meta, iteration)
            loss_rec = meta['loss_rec']
            loss_kl = meta['loss_kl']

            print(f'AE:[{iteration + 1}/{n_iterations}], loss_rec={loss_rec:.2f}, '
                    f'loss_kl={loss_kl:.2f}, lambda={lambda_kl:.1e}     ', end='\r')
            if (iteration + 1) % self.args.save_steps == 0 or iteration + 1 == n_iterations:
                self.save_model(iteration=iteration)
                print()
        return

