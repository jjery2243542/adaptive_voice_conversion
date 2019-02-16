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
import json
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from scipy.io.wavfile import write
import random

class Evaluater(object):
    def __init__(self, config, args):
        # config store the value of hyperparameters, turn to attr by AttrDict
        self.config = config
        print(config)

        # args store other information
        self.args = args
        print(self.args)

        # get dataloader
        self.load_data()

        # init the model with config
        self.build_model()

        # load model
        self.load_model()
        # read speaker info
        self.speaker2gender = self.read_speaker_gender(self.args.speaker_info_path)
        # sampled n speakers for evaluation
        self.sample_n_speakers(self.args.n_speakers)
        # read mean, std
        self.mean, self.std = self.read_norm_param()

    def read_norm_param(self):
        with open(os.path.join(self.args.data_dir, self.args.norm_param_file)) as f:
            data = json.load(f)
        return data['mean'], data['std']

    def load_model(self, load_opt=False, load_dis=False):
        print(f'Load model from {self.args.load_model_path}, load_opt={load_opt}, load_dis={load_dis}')
        self.model.load_state_dict(torch.load(f'{self.args.load_model_path}.ckpt'))
        if load_dis:
            self.discr.load_state_dict(torch.load(f'{self.args.load_model_path}.discr'))
        if load_opt:
            self.gen_opt.load_state_dict(torch.load(f'{self.args.load_model_path}.opt'))
        if load_dis and load_opt:
            self.dis_opt.load_state_dict(torch.load(f'{self.args.load_model_path}.discr.opt'))
        return

    def load_data(self):
        data_dir = self.args.data_dir

        # load pkl data and sampled segments
        with open(os.path.join(data_dir, f'{self.args.val_set}.pkl'), 'rb') as f:
            self.pkl_data = pickle.load(f)
        with open(os.path.join(data_dir, self.args.val_index_file), 'r') as f:
            self.indexes = json.load(f)
        return

    def build_model(self): 
        # create model, discriminator, optimizers
        self.model = cc(AE(c_in=self.config.c_in,
                c_h=self.config.c_h,
                c_latent=self.config.c_latent,
                c_cond=self.config.c_cond,
                c_out=self.config.c_in,
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
                dropout_rate=self.config.dropout_rate))
        print(self.model)
        self.model.eval()
        self.noise_adder = NoiseAdder(0, self.config.gaussian_std)
        return

    def sample_n_speakers(self, n_speakers):
        # only apply on VCTK corpus
        self.speakers = sorted(list(set([key.split('_')[0] for key in self.pkl_data])))
        # first n speakers are sampled
        self.sampled_speakers = self.speakers[:n_speakers]
        self.speaker_index = {speaker:i for i, speaker in enumerate(self.sampled_speakers)}
        return

    def read_speaker_gender(self, speaker_path):
        speaker2gender = {}
        with open(speaker_path, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                id, _, gender, _ = line.strip().split(maxsplit=3)
                speaker2gender[f'p{id}'] = gender
        return speaker2gender

    def plot_static_embeddings(self, output_path):
        # filter the samples by speakers sampled
        # hack code 
        small_indexes = [index for index in self.indexes if index[0][:len('p000')] in self.sampled_speakers]
        random.shuffle(small_indexes)
        small_indexes = small_indexes[:self.args.max_samples]
        # generate the tensor and dataloader for evaluation
        tensor = [self.pkl_data[key][t:t + self.config.segment_size] for key, t, _, _, _ in small_indexes]
        speakers = [key[:len('p000')] for key, _, _, _, _  in small_indexes]
        # add the dimension for channel
        tensor = torch.from_numpy(np.array(tensor)).unsqueeze(dim=1)
        dataset = TensorDataset(tensor)
        dataloader = DataLoader(dataset, batch_size=20, shuffle=False, num_workers=0)
        all_embs = []
        # run the model 
        for data in dataloader:
            data = cc(data[0])
            embs = self.model.get_static_embeddings(data)
            all_embs = all_embs + embs.detach().cpu().numpy().tolist()
        all_embs = np.array(all_embs)
        print(all_embs.shape)
        # TSNE
        embs_2d = TSNE(n_components=2, init='pca', perplexity=50).fit_transform(all_embs)
        x_min, x_max = embs_2d.min(0), embs_2d.max(0)
        embs_norm = (embs_2d - x_min) / (x_max - x_min)
        # plot to figure
        female_cluster = [i for i, speaker in enumerate(speakers) if self.speaker2gender[speaker] == 'F']
        male_cluster = [i for i, speaker in enumerate(speakers) if self.speaker2gender[speaker] == 'M']
        colors = np.array([self.speaker_index[speaker] for speaker in speakers])
        plt.scatter(embs_norm[female_cluster, 0], embs_norm[female_cluster, 1], 
                c=colors[female_cluster], marker='x') 
        plt.scatter(embs_norm[male_cluster, 0], embs_norm[male_cluster, 1], 
                c=colors[male_cluster], marker='o') 
        plt.savefig(output_path)
        return

    def inference_one_utterance(self, x, x_cond, output_path):
        x = x.view(1, 1, x.size(0))
        x_cond = x_cond.view(1, 1, x_cond.size(0))
        dec = self.model.inference(x, x_cond)
        dec = dec * self.std + self.mean
        dec = dec.detach().cpu().numpy()
        write(output_path, rate=22050, data=dec)
        return

    def infer_default(self):
        # using the first sample from in_test
        content_utt, _, _, cond_utt, _ = self.indexes[0]
        content = torch.from_numpy(self.pkl_data[content_utt]).cuda()
        cond = torch.from_numpy(self.pkl_data[cond_utt]).cuda()
        self.inference_one_utterance(content, cond, f'{args.output_path}.src2tar.wav')
        # reconstruction
        self.inference_one_utterance(content, content, f'{args.output_path}.rec_src.wav')
        self.inference_one_utterance(cond, cond, f'{args.output_path}.rec_tar.wav')
        return


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-config', '-c', default='config.yaml')
    parser.add_argument('-data_dir', '-d', 
            default='/storage/feature/voice_conversion/trimmed_vctk_waveform/librosa/split_10_0.1')
    parser.add_argument('-val_set', default='in_test')
    parser.add_argument('-val_index_file', default='in_test_samples_4000.json')
    parser.add_argument('-norm_param_file', default='mean_std.json')
    parser.add_argument('-load_model_path', default='/storage/model/adaptive_vc/model')
    parser.add_argument('--plot_speakers', action='store_true')
    parser.add_argument('-fig_output_path', default='speaker.png')
    parser.add_argument('-n_speakers', default=8, type=int)
    parser.add_argument('-speaker_info_path', default='/storage/datasets/VCTK/VCTK-Corpus/speaker-info.txt')
    parser.add_argument('-max_samples', default=1000, type=int)
    parser.add_argument('--infer_default', action='store_true')
    parser.add_argument('-output_path', default='test')

    args = parser.parse_args()
    # load config file 
    with open(args.config) as f:
        config = yaml.load(f)
        config = Namespace(**config)
    evaluator = Evaluater(config=config, args=args)

    if args.plot_speakers:
        evaluator.plot_static_embeddings(args.fig_output_path)

    if args.infer_default:
        evaluator.infer_default()
