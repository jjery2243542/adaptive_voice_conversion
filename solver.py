import torch
import numpy as np
import sys
import os 
import torch.nn as nn
import torch.nn.functional as F
import yaml
import pickle
from model import AE, LatentDiscriminator, ProjectionDiscriminator, cal_gradpen, compute_grad
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
        torch.save(self.gen_opt.state_dict(), f'{self.args.store_model_path}.{stage}.gen.opt')
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
            self.gen_opt.load_state_dict(torch.load(f'{self.args.load_model_path}.gen.opt'))
        if load_dis and load_opt:
            self.la_dis_opt.load_state_dict(torch.load(f'{self.args.load_model_path}.discr.opt'))
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
                act=self.config.act,
                dropout_rate=self.config.dropout_rate))
        print(self.model)
        self.discr = cc(ProjectionDiscriminator(
            input_size=(self.config.c_in, self.config.segment_size),
            c_in=1, 
            c_h=self.config.dis_c_h, 
            c_cond=self.config.c_cond, 
            kernel_size=self.config.dis_kernel_size, 
            n_conv_blocks=self.config.dis_n_conv_blocks, 
            n_dense_layers=self.config.dis_n_dense_layers, 
            d_h=self.config.dis_d_h, act=self.config.act, sn=self.config.sn,
            sim_layer=self.config.sim_layer))
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
            enc, emb, emb_pos, dec = self.model(self.noise_adder(x), 
                    self.noise_adder(x_pos), 
                    self.noise_adder(x_neg), 
                    mode='pretrain_ae')
        else:
            enc, emb, emb_pos, dec = self.model(x, 
                    x_pos, 
                    x_neg,
                    mode='pretrain_ae')

        loss_rec = self.weighted_l1_loss(dec, x)
        loss_kl = torch.mean(enc ** 2)
        loss_sim = torch.mean((emb - emb_pos) ** 2)
        loss = lambda_rec * loss_rec + \
                self.config.lambda_kl * loss_kl + \
                self.config.lambda_sim * loss_sim
        self.ae_opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_norm)
        self.ae_opt.step()
        meta = {'loss_rec': loss_rec.item(),
                'loss_kl': loss_kl.item(),
                'loss_sim': loss_sim.item(),
                'grad_norm': grad_norm}
        return meta

    def ae_gen_step(self, data, lambda_dis):
        x, x_pos, x_neg = [cc(tensor) for tensor in data]
        if self.config.add_gaussian:
            enc, enc_pos, emb_pos, emb_neg, emb_rec, dec, dec_syn = self.model(self.noise_adder(x), 
                    self.noise_adder(x_pos), 
                    self.noise_adder(x_neg),
                    mode='gen_ae')
        else:
            enc, enc_pos, emb_pos, emb_neg, emb_rec, dec, dec_syn = self.model(x, 
                    x_pos, 
                    x_neg, 
                    mode='gen_ae')

        loss_srec = torch.mean(torch.abs(emb_neg - emb_rec))

        _, real_h = self.discr(x)
        _, fake_h = self.discr(dec)
        loss_rec = torch.mean(torch.abs(real_h - fake_h))
        loss_kl = torch.mean(enc ** 2)

        fake_vals, _ = self.discr(dec_syn, emb_neg.detach())
        criterion = nn.BCEWithLogitsLoss()
        ones_label = fake_vals.new_ones(*fake_vals.size())
        loss_dis = criterion(fake_vals, ones_label)

        loss = self.config.lambda_lsm_rec * loss_rec + \
                self.config.lambda_kl * loss_kl + \
                self.config.lambda_srec * loss_srec + \
                lambda_dis * loss_dis 

        self.ae_opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_norm)
        self.ae_opt.step()

        meta = {'loss_rec': loss_rec.item(),
                'loss_srec': loss_srec.item(),
                'loss_dis': loss_dis.item(),
                'loss': loss.item(), 
                'grad_norm': grad_norm}
        return meta

    def dis_step(self, data_real, data_gen):
        x, _, _ = [cc(tensor) for tensor in data_real]
        x_prime, x_pos, x_neg = [cc(tensor) for tensor in data_gen]

        with torch.no_grad():
            if self.config.add_gaussian:
                emb = self.model(self.noise_adder(x), x_pos=None, x_neg=None, mode='dis_real')
                _, _, emb_pos, emb_neg, dec, dec_syn = self.model(self.noise_adder(x_prime), 
                        x_pos=self.noise_adder(x_pos), 
                        x_neg=self.noise_adder(x_neg), 
                        mode='dis_fake')
            else:
                emb = self.model(x, x_pos=None, x_neg=None, mode='dis_real')
                _, _, emb_pos, emb_neg, dec, dec_syn = self.model(x_prime, 
                        x_pos=x_pos, 
                        x_neg=x_neg, 
                        mode='dis_fake')
        # for R1 regularization
        x.requires_grad = True
        # input for the discriminator
        real_vals, _ = self.discr(x, emb)
        rec_fake_vals, _ = self.discr(dec, emb_pos)
        con_fake_vals, _ = self.discr(dec_syn, emb_neg)

        ones_label = real_vals.new_ones(*real_vals.size()) * self.config.one_side_ls_weight 
        zeros_label = rec_fake_vals.new_zeros(*rec_fake_vals.size())
        criterion = nn.BCEWithLogitsLoss()

        loss_real = criterion(real_vals, ones_label)
        loss_rec_fake = criterion(rec_fake_vals, zeros_label)
        loss_con_fake = criterion(con_fake_vals, zeros_label)

        loss_gp = compute_grad(real_vals, x)
        loss_dis = loss_real + (loss_rec_fake + loss_con_fake) / 2
        loss = loss_dis + self.config.lambda_gp * loss_gp

        self.dis_opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.discr.parameters(), max_norm=self.config.grad_norm)
        self.dis_opt.step()

        real_probs = torch.sigmoid(real_vals)
        rec_fake_probs = torch.sigmoid(rec_fake_vals)
        con_fake_probs = torch.sigmoid(con_fake_vals)

        acc_real = torch.mean((real_probs >= 0.5).float())
        acc_rec_fake = torch.mean((rec_fake_probs < 0.5).float())
        acc_con_fake = torch.mean((con_fake_probs < 0.5).float())
        acc = (acc_real + acc_rec_fake + acc_con_fake) / 3

        meta = {'loss_dis': loss_dis.item(),
                'loss_real': loss_real.item(),
                'loss_rec_fake': loss_rec_fake.item(),
                'loss_con_fake': loss_con_fake.item(),
                'loss_gp': loss_gp.item(),
                'real_prob': torch.mean(real_probs).item(),
                'rec_fake_prob': torch.mean(rec_fake_probs).item(),
                'con_fake_prob': torch.mean(con_fake_probs).item(),
                'acc_real': acc_real.item(),
                'acc_rec_fake': acc_rec_fake.item(),
                'acc_con_fake': acc_con_fake.item(),
                'acc': acc.item(), 
                'grad_norm': grad_norm}
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
            loss_sim = meta['loss_sim']
            loss_kl = meta['loss_kl']

            print(f'AE:[{iteration + 1}/{n_iterations}], loss_rec={loss_rec:.2f}, '
                    f'loss_kl={loss_kl:.2f}, '
                    f'loss_sim={loss_sim:.2f}, '
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
                data = next(self.train_iter)
                gen_meta = self.ae_gen_step(data, lambda_dis=lambda_dis)
                # add to logger
                if iteration % self.args.summary_steps == 0:
                    self.logger.scalars_summary(f'{self.args.tag}/ae_gen_train', gen_meta, iteration) 

            # D step
            for dis_step in range(self.config.dis_steps):
                data, data_prime = next(self.train_iter), next(self.train_iter)
                dis_meta = self.dis_step(data, data_prime)
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
            data, data_prime = next(self.train_iter), next(self.train_iter)
            meta = self.dis_step(data, data_prime)
            # add to logger
            if iteration % self.args.summary_steps == 0:
                self.logger.scalars_summary(f'{self.args.tag}/dis_pretrain', meta, iteration)

            real_prob = meta['real_prob']
            fake_prob = meta['con_fake_prob']
            gp = meta['loss_gp']

            print(f'D:[{iteration + 1}/{n_iterations}], '
                    f'real_prob={real_prob:.2f}, fake_prob={fake_prob:.2f}, gp={gp:.2f}     ', end='\r')

            if (iteration + 1) % self.args.save_steps == 0 or iteration + 1 == n_iterations:
                self.save_model(iteration=iteration, stage='dis')
                print()
        return
