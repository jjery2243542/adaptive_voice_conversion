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

    def weighted_loss(self, dec, x):
        criterion = nn.L1Loss()
        n_priority_freq = int(3000 / (self.config['sample_rate'] * 0.5) * self.config['ContentEncoder']['c_in'])
        loss_rec = 0.5 * criterion(dec, x) + 0.5 * criterion(dec[:, :n_priority_freq], x[:, :n_priority_freq])
        return loss_rec

    def ae_step(self, data, lambda_kl):
        x, _ = [cc(tensor) for tensor in data]
        enc, emb, dec = self.model(x)
        #loss_rec = self.weighted_loss(dec, x)
        criterion = nn.L1Loss()
        loss_rec = criterion(dec, x)
        loss_kl = torch.mean(enc ** 2)
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
                    f'loss_kl={loss_kl:.2f}     ', end='\r')
            if (iteration + 1) % self.args.save_steps == 0 or iteration + 1 == n_iterations:
                self.save_model(iteration=iteration)
                print()
        return

'''
    def ae_gen_step(self, data_ae, data_gen, lambda_dis):
        x, _ = [cc(tensor) for tensor in data_ae]
        x_prime, x_neg = [cc(tensor) for tensor in data_gen]
        enc, emb, dec = self.model(x, x_neg=None, mode='pretrain_ae')
        _, emb_neg, emb_rec, dec_syn = self.model(x_prime, x_neg=x_neg, mode='gen_ae')

        criterion = nn.L1Loss()
        loss_rec = criterion(dec, x)
        loss_srec = criterion(emb_neg, emb_rec)
        fake_vals, cond_vals = self.discr(dec_syn, emb_neg.detach())
        loss_dis = -torch.mean(fake_vals)

        loss = self.config.final_lambda_rec * loss_rec + \
                self.config.lambda_srec * loss_srec + \
                lambda_dis * loss_dis 

        self.gen_opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_norm)
        self.gen_opt.step()
        for name, param in self.model.decoder.named_parameters():
            if param.requires_grad:
                param.data = self.ema(name, param.data)

        meta = {'loss_rec': loss_rec.item(),
                'loss_srec': loss_srec.item(),
                'loss_dis': loss_dis.item(),
                'loss': loss.item(),
                'cond_val': torch.mean(cond_vals).item(),
                'grad_norm': grad_norm}
        return meta

    def dis_step(self, data_real, data_fake, data_mismatch):
        x, _ = [cc(tensor) for tensor in data_real]
        x_prime, x_neg = [cc(tensor) for tensor in data_fake]
        x_mismatch, x_mismatch_neg = [cc(tensor) for tensor in data_mismatch]

        with torch.no_grad():
            emb = self.model(x, x_neg=None, mode='dis_real')
            _, emb_syn, dec_syn = self.model(x=x_prime, x_neg=x_neg, mode='dis_fake')
            if self.config.use_mismatch:
                emb_neg = self.model(x=None, x_neg=x_mismatch_neg, mode='dis_mismatch')

        # for R1 regularization
        x.requires_grad = True
        emb.requires_grad = True
        if self.config.instance_noise:
            x = self.noise_adder(x)
            dec_syn = self.noise_adder(dec_syn)
        # input for the discriminator
        real_vals, real_cond_vals = self.discr(x, emb)
        fake_vals, fake_cond_vals = self.discr(dec_syn, emb_syn)

        loss_real = torch.mean(F.relu(1.0 - real_vals))
        loss_fake = torch.mean(F.relu(1.0 + fake_vals))

        if self.config.use_mismatch:
            mismatch_vals, mismatch_cond_vals = self.discr(x_mismatch, emb_neg)
            loss_mismatch = torch.mean(F.relu(1.0 + mismatch_vals))
            loss_dis = loss_real + (loss_fake + loss_mismatch) / 2
        else:
            loss_dis = loss_real + loss_fake
        loss = loss_dis
        if self.config.lambda_gp > 0:
            loss_gp = compute_grad(real_vals, x)
            loss += loss_gp

        self.dis_opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.discr.parameters(), max_norm=self.config.grad_norm)
        self.dis_opt.step()

        meta = {'loss_dis': loss_dis.item(),
                'loss_real': loss_real.item(),
                'loss_fake': loss_fake.item(),
                'real_val': torch.mean(real_vals).item(),
                'fake_val': torch.mean(fake_vals).item(),
                'real_cond_val': torch.mean(real_cond_vals).item(),
                'fake_cond_val': torch.mean(fake_cond_vals).item(),
                'grad_norm': grad_norm}
        if self.config.use_mismatch:
            meta['mismatch_val'] = torch.mean(mismatch_vals).item()
            meta['mismatch_cond_val'] = torch.mean(mismatch_cond_vals).item()
            meta['loss_mismatch'] = loss_mismatch.item()
        if self.config.lambda_gp > 0:
            meta['loss_gp'] = loss_gp.item()
        return meta
    def ae_gen_train(self, n_iterations):
        for name, param in self.model.decoder.named_parameters():
            if param.requires_grad:
                self.ema.register(name, param.data)
        for iteration in range(n_iterations):
            # calculate linear increasing lambda
            if iteration >= self.config.dis_sched_iters:
                lambda_dis = self.config.lambda_dis
            else:
                lambda_dis = self.config.lambda_dis * (iteration + 1) / self.config.dis_sched_iters
            # AE step
            for ae_step in range(self.config.ae_steps):
                data_ae, data_gen = next(self.train_iter), next(self.train_iter)
                gen_meta = self.ae_gen_step(data_ae, data_gen, lambda_dis=lambda_dis)
                # add to logger
                if iteration % self.args.summary_steps == 0:
                    self.logger.scalars_summary(f'{self.args.tag}/gen_train', gen_meta, iteration) 

            # D step
            for dis_step in range(self.config.dis_steps):
                data_real = next(self.train_iter)
                data_fake, data_mismatch = next(self.train_iter), next(self.train_iter)
                dis_meta = self.dis_step(data_real, data_fake, data_mismatch)
                # add to logger
                if iteration % self.args.summary_steps == 0:
                    self.logger.scalars_summary(f'{self.args.tag}/dis_train', dis_meta, iteration) 

            loss_rec = gen_meta['loss_rec']
            loss_srec = gen_meta['loss_srec']
            loss_dis = gen_meta['loss_dis']
            print(f'G:[{iteration + 1}/{n_iterations}], loss_rec={loss_rec:.2f}, '
                    f'loss_srec={loss_srec:.2f}, '
                    f'loss_dis={loss_dis:.2f}, '
                    f'lambda={lambda_dis:.1e}     ', 
                    end='\r')
            if (iteration + 1) % self.args.save_steps == 0 or iteration + 1 == n_iterations:
                print()
                self.save_model(iteration=iteration, stage='gen')

    def dis_pretrain(self, n_iterations):
        for iteration in range(n_iterations):
            data_real = next(self.train_iter)
            data_fake, data_mismatch = next(self.train_iter), next(self.train_iter)
            meta = self.dis_step(data_real, data_fake, data_mismatch)
            # add to logger
            if iteration % self.args.summary_steps == 0:
                self.logger.scalars_summary(f'{self.args.tag}/dis_pretrain', meta, iteration)

            real_val = meta['real_val']
            fake_val = meta['fake_val']

            print(f'D:[{iteration + 1}/{n_iterations}], '
                    f'real_val={real_val:.2f}, '
                    f'fake_val={fake_val:.2f}      ', end='\r')

            if (iteration + 1) % self.args.save_steps == 0 or iteration + 1 == n_iterations:
                self.save_model(iteration=iteration, stage='dis')
                print()
        return
'''
