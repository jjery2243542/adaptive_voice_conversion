import torch
import numpy as np
import sys
import os 
import torch.nn as nn
import torch.nn.functional as F
import yaml
import pickle
from model import AE, Discriminator
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
            self.load_model(args.load_opt, args.load_dis)

    def save_model(self, iteration, stage):
        # save model and discriminator and their optimizer
        torch.save(self.model.state_dict(), f'{self.args.store_model_path}.{stage}.ckpt')
        torch.save(self.ae_opt.state_dict(), f'{self.args.store_model_path}.{stage}.opt')
        torch.save(self.discr.state_dict(), f'{self.args.store_model_path}.{stage}.discr')
        torch.save(self.dis_opt.state_dict(), f'{self.args.store_model_path}.{stage}.discr.opt')

    def save_config(self):
        with open(f'{self.args.store_model_path}.config.yaml', 'w') as f:
            yaml.dump(vars(self.config), f)
        with open(f'{self.args.store_model_path}.args.yaml', 'w') as f:
            yaml.dump(vars(self.args), f)
        return

    def load_model(self, load_opt, load_dis):
        print(f'Load model from {self.args.load_model_path}, load_opt={load_opt}, load_dis={load_dis}')
        self.model.load_state_dict(torch.load(f'{self.args.load_model_path}.ckpt'))
        if load_dis:
            self.discr.load_state_dict(torch.load(f'{self.args.load_model_path}.discr'))
        if load_opt:
            self.ae_opt.load_state_dict(torch.load(f'{self.args.load_model_path}.opt'))
        if load_dis and load_opt:
            self.dis_opt.load_state_dict(torch.load(f'{self.args.load_model_path}.discr.opt'))
        return

    def get_data_loaders(self):
        data_dir = self.args.data_dir

        self.train_dataset = PickleDataset(os.path.join(data_dir, f'{self.args.train_set}.pkl'), 
                os.path.join(data_dir, self.args.train_index_file), 
                segment_size=self.config.segment_size)
        self.train_loader = get_data_loader(self.train_dataset,
                frame_size=self.config.frame_size,
                batch_size=self.config.batch_size, 
                shuffle=self.config.shuffle, 
                num_workers=4, drop_last=False)
        self.train_iter = infinite_iter(self.train_loader)
        return

    def build_model(self): 
        # create model, discriminator, optimizers
        self.model = cc(AE(input_size=self.config.segment_size // self.config.frame_size,
                c_in=self.config.c_in * self.config.frame_size,
                s_c_h=self.config.s_c_h,
                d_c_h=self.config.d_c_h,
                c_latent=self.config.c_latent,
                c_cond=self.config.c_cond,
                c_out=self.config.c_in * self.config.frame_size,
                c_bank=self.config.c_bank,
                bank_size=self.config.bank_size,
                bank_scale=self.config.bank_scale,
                kernel_size=self.config.kernel_size,
                s_enc_n_conv_blocks=self.config.s_enc_n_conv_blocks,
                s_enc_n_dense_blocks=self.config.s_enc_n_dense_blocks,
                d_enc_n_conv_blocks=self.config.d_enc_n_conv_blocks,
                d_enc_n_dense_blocks=self.config.d_enc_n_dense_blocks,
                s_subsample=self.config.s_subsample,
                d_subsample=self.config.d_subsample,
                dec_n_conv_blocks=self.config.dec_n_conv_blocks,
                dec_n_dense_blocks=self.config.dec_n_dense_blocks,
                dec_n_mlp_blocks=self.config.dec_n_mlp_blocks,
                upsample=self.config.upsample,
                act=self.config.gen_act,
                dropout_rate=self.config.dropout_rate, use_dummy=self.config.use_dummy, sn=self.config.sn))
        print(self.model)
        self.discr = cc(Discriminator(
            input_size=(self.config.c_in, self.config.segment_size),
            c_in=1, 
            c_h=self.config.dis_c_h, 
            c_cond=self.config.c_cond,
            subsample=self.config.dis_subsample,
            kernel_size=self.config.dis_kernel_size, 
            n_conv_blocks=self.config.dis_n_conv_blocks, 
            n_dense_layers=self.config.dis_n_dense_layers, 
            d_h=self.config.dis_d_h, act=self.config.dis_act, sn=self.config.sn))
        print(self.discr)
        self.ae_opt = torch.optim.Adam(self.model.parameters(), 
                lr=self.config.gen_lr, betas=(self.config.beta1, self.config.beta2), 
                amsgrad=self.config.amsgrad, weight_decay=self.config.weight_decay)
        self.gen_opt = torch.optim.Adam(self.model.decoder.parameters(), 
                lr=self.config.gen_lr, betas=(self.config.beta1, self.config.beta2), 
                amsgrad=self.config.amsgrad, weight_decay=self.config.weight_decay)
        self.dis_opt = torch.optim.Adam(self.discr.parameters(), 
                lr=self.config.dis_lr, betas=(self.config.beta1, self.config.beta2), 
                amsgrad=self.config.amsgrad, weight_decay=self.config.weight_decay)  
        print(self.ae_opt)
        print(self.gen_opt)
        print(self.dis_opt)
        self.noise_adder = NoiseAdder(0, self.config.gaussian_std)
        return

    def weighted_l1_loss(self, dec, x):
        criterion = nn.L1Loss()
        n_priority_freq = int(3000 / (self.config.sample_rate * 0.5) * self.config.c_in)
        loss_rec = 0.5 * criterion(dec, x) + 0.5 * criterion(dec[:, :n_priority_freq], x[:, :n_priority_freq])
        return loss_rec

    def ae_pretrain_step(self, data, lambda_rec):
        x, x_pos, x_neg = [cc(tensor) for tensor in data]
        if self.config.add_gaussian:
            enc, emb_pos, dec = self.model(self.noise_adder(x), 
                    self.noise_adder(x_pos), 
                    self.noise_adder(x_neg), 
                    mode='pretrain_ae')
        else:
            enc, emb_pos, dec = self.model(x, 
                    x_pos, 
                    x_neg,
                    mode='pretrain_ae')

        loss_rec = self.weighted_l1_loss(dec, x)
        loss_kl = torch.mean(enc ** 2)
        loss = lambda_rec * loss_rec + \
                self.config.lambda_kl * loss_kl
        self.ae_opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_norm)
        self.ae_opt.step()
        meta = {'loss_rec': loss_rec.item(),
                'loss_kl': loss_kl.item(),
                'grad_norm': grad_norm}
        return meta

    def ae_gen_step(self, data_ae, data_gen, lambda_dis):
        x, x_pos, _ = [cc(tensor) for tensor in data_ae]
        x_prime, _, x_neg = [cc(tensor) for tensor in data_ae]
        if self.config.add_gaussian:
            enc, emb_pos, dec = self.model(self.noise_adder(x), 
                    self.noise_adder(x_pos), 
                    x_neg=None, 
                    mode='pretrain_ae')
            _, emb_neg, emb_rec, dec_syn = self.model(self.noise_adder(x_prime), 
                    x_pos=None, 
                    x_neg=self.noise_adder(x_neg),
                    mode='gen_ae')
        else:
            enc, emb_pos, dec = self.model(x, 
                    x_pos, 
                    x_neg=None, 
                    mode='pretrain_ae')
            _, emb_neg, emb_rec, dec_syn = self.model(x_prime, 
                    x_pos=None, 
                    x_neg=x_neg,
                    mode='gen_ae')

        loss_rec = torch.mean(torch.abs(dec - x))
        loss_srec = torch.mean(torch.abs(emb_neg - emb_rec))
        fake_vals = self.discr(dec_syn, emb_neg.detach())
        loss_dis = -torch.mean(fake_vals)

        loss = self.config.final_lambda_rec * loss_rec + \
                self.config.lambda_srec * loss_srec + \
                lambda_dis * loss_dis 

        self.gen_opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_norm)
        self.gen_opt.step()

        meta = {'loss_rec': loss_rec.item(),
                'loss_srec': loss_srec.item(),
                'loss_dis': loss_dis.item(),
                'loss': loss.item(), 
                'grad_norm': grad_norm}
        return meta

    def dis_step(self, data_real, data_fake, data_mismatch):
        x, _, _ = [cc(tensor) for tensor in data_real]
        x_prime, _, x_neg = [cc(tensor) for tensor in data_fake]
        x_mismatch, _, x_mismatch_neg = [cc(tensor) for tensor in data_mismatch]

        with torch.no_grad():
            if self.config.add_gaussian:
                emb = self.model(self.noise_adder(x), x_pos=None, x_neg=None, mode='dis_real')
                _, emb_syn, dec_syn = self.model(self.noise_adder(x_prime), 
                        x_pos=None, 
                        x_neg=self.noise_adder(x_neg), 
                        mode='dis_fake')
                if self.config.use_mismatch:
                    emb_neg = self.model(x=None, 
                            x_pos=None, x_neg=self.noise_adder(x_mismatch_neg), mode='dis_mismatch') 
            else:
                emb = self.model(x, x_pos=None, x_neg=None, mode='dis_real')
                _, emb_syn, dec_syn = self.model(x=x_prime, 
                        x_pos=None, 
                        x_neg=x_neg, 
                        mode='dis_fake')
                if self.config.use_mismatch:
                    emb_neg = self.model(x=None, 
                            x_pos=None, x_neg=x_mismatch_neg, mode='dis_mismatch')

        # for R1 regularization
        x.requires_grad = True
        emb.requires_grad = True
        # input for the discriminator
        real_vals = self.discr(x, emb)
        fake_vals = self.discr(dec_syn, emb_syn)

        loss_real = torch.mean(F.relu(1.0 - real_vals))
        loss_fake = torch.mean(F.relu(1.0 + fake_vals))

        if self.config.use_mismatch:
            mismatch_vals = self.discr(x_mismatch, emb_neg)
            loss_mismatch = torch.mean(F.relu(1.0 + mismatch_vals))
            loss_dis = loss_real + (loss_fake + loss_mismatch) / 2
        else:
            loss_dis = loss_real + loss_fake
        loss = loss_dis

        self.dis_opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.discr.parameters(), max_norm=self.config.grad_norm)
        self.dis_opt.step()


        meta = {'loss_dis': loss_dis.item(),
                'loss_real': loss_real.item(),
                'loss_fake': loss_fake.item(),
                'real_val': torch.mean(real_vals).item(),
                'fake_val': torch.mean(fake_vals).item(),
                'grad_norm': grad_norm}
        if self.config.use_mismatch:
            meta['mismatch_val'] = torch.mean(mismatch_vals).item()
            meta['loss_mismatch'] = loss_mismatch.item()
        return meta

    def ae_pretrain(self, n_iterations):
        for iteration in range(n_iterations):
            if iteration >= self.config.rec_sched_iters:
                lambda_rec = self.config.final_lambda_rec
            else:
                lambda_rec = self.config.init_lambda_rec + \
                        (self.config.final_lambda_rec - self.config.init_lambda_rec) * \
                        (iteration + 1) / self.config.rec_sched_iters
            data = next(self.train_iter)
            meta = self.ae_pretrain_step(data, lambda_rec)
            # add to logger
            if iteration % self.args.summary_steps == 0:
                self.logger.scalars_summary(f'{self.args.tag}/ae_pretrain', meta, iteration)
            loss_rec = meta['loss_rec']
            loss_kl = meta['loss_kl']

            print(f'AE:[{iteration + 1}/{n_iterations}], loss_rec={loss_rec:.2f}, '
                    f'loss_kl={loss_kl:.2f}, '
                    f'lambda={lambda_rec:.1e}     ', 
                    end='\r')
            if (iteration + 1) % self.args.save_steps == 0 or iteration + 1 == n_iterations:
                self.save_model(iteration=iteration, stage='ae')
                print()
        return

    def ae_gen_train(self, n_iterations):
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
