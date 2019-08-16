import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
import numpy as np
from math import ceil
from functools import reduce
from torch.nn.utils import spectral_norm
from utils import cc

class DummyEncoder(object):
    def __init__(self, encoder):
        self.encoder = encoder

    def load(self, target_network):
        self.encoder.load_state_dict(target_network.state_dict())

    def __call__(self, x):
        return self.encoder(x)

def pad_layer(inp, layer, pad_type='reflect'):
    kernel_size = layer.kernel_size[0]
    if kernel_size % 2 == 0:
        pad = (kernel_size//2, kernel_size//2 - 1)
    else:
        pad = (kernel_size//2, kernel_size//2)
    # padding
    inp = F.pad(inp, 
            pad=pad,
            mode=pad_type)
    out = layer(inp)
    return out

def pad_layer_2d(inp, layer, pad_type='reflect'):
    kernel_size = layer.kernel_size
    if kernel_size[0] % 2 == 0:
        pad_lr = [kernel_size[0]//2, kernel_size[0]//2 - 1]
    else:
        pad_lr = [kernel_size[0]//2, kernel_size[0]//2]
    if kernel_size[1] % 2 == 0:
        pad_ud = [kernel_size[1]//2, kernel_size[1]//2 - 1]
    else:
        pad_ud = [kernel_size[1]//2, kernel_size[1]//2]
    pad = tuple(pad_lr + pad_ud)
    # padding
    inp = F.pad(inp, 
            pad=pad,
            mode=pad_type)
    out = layer(inp)
    return out

def pixel_shuffle_1d(inp, scale_factor=2):
    batch_size, channels, in_width = inp.size()
    channels //= scale_factor
    out_width = in_width * scale_factor
    inp_view = inp.contiguous().view(batch_size, channels, scale_factor, in_width)
    shuffle_out = inp_view.permute(0, 1, 3, 2).contiguous()
    shuffle_out = shuffle_out.view(batch_size, channels, out_width)
    return shuffle_out

def upsample(x, scale_factor=2):
    x_up = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
    return x_up

def flatten(x):
    out = x.contiguous().view(x.size(0), x.size(1) * x.size(2))
    return out

def concat_cond(x, cond):
    # x = [batch_size, x_channels, length]
    # cond = [batch_size, c_channels]
    cond = cond.unsqueeze(dim=2)
    cond = cond.expand(*cond.size()[:-1], x.size(-1))
    out = torch.cat([x, cond], dim=1)
    return out

def append_cond(x, cond):
    # x = [batch_size, x_channels, length]
    # cond = [batch_size, x_channels * 2]
    p = cond.size(1) // 2
    mean, std = cond[:, :p], cond[:, p:]
    out = x * std.unsqueeze(dim=2) + mean.unsqueeze(dim=2)
    return out

def conv_bank(x, module_list, act, pad_type='reflect'):
    outs = []
    for layer in module_list:
        out = act(pad_layer(x, layer, pad_type))
        outs.append(out)
    out = torch.cat(outs + [x], dim=1)
    return out

def get_act(act):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'lrelu':
        return nn.LeakyReLU()
    else:
        return nn.ReLU()

class MLP(nn.Module):
    def __init__(self, c_in, c_h, n_blocks, act, sn):
        super(MLP, self).__init__()
        self.act = get_act(act)
        self.n_blocks = n_blocks
        f = spectral_norm if sn else lambda x: x
        self.in_dense_layer = f(nn.Linear(c_in, c_h))
        self.first_dense_layers = nn.ModuleList([f(nn.Linear(c_h, c_h)) for _ in range(n_blocks)])
        self.second_dense_layers = nn.ModuleList([f(nn.Linear(c_h, c_h)) for _ in range(n_blocks)])

    def forward(self, x):
        h = self.in_dense_layer(x)
        for l in range(self.n_blocks):
            y = self.first_dense_layers[l](h)
            y = self.act(y)
            y = self.second_dense_layers[l](y)
            y = self.act(y)
            h = h + y
        return h

class Prenet(nn.Module):
    def __init__(self, c_in, c_h, c_out, 
            kernel_size, n_conv_blocks, 
            subsample, act, dropout_rate):
        super(Prenet, self).__init__()
        self.act = get_act(act)
        self.subsample = subsample
        self.n_conv_blocks = n_conv_blocks
        self.in_conv_layer = nn.Conv2d(1, c_h, kernel_size=kernel_size)
        self.first_conv_layers = nn.ModuleList([nn.Conv2d(c_h, c_h, kernel_size=kernel_size) for _ \
                in range(n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList([nn.Conv2d(c_h, c_h, kernel_size=kernel_size, stride=sub) 
            for sub, _ in zip(subsample, range(n_conv_blocks))])
        output_size = c_in
        for l, sub in zip(range(n_conv_blocks), self.subsample):
            output_size = ceil(output_size / sub)
        self.out_conv_layer = nn.Conv1d(c_h * output_size, c_out, kernel_size=1)
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.norm_layer = nn.InstanceNorm2d(c_h, affine=False)

    def forward(self, x):
        # reshape x to 4D
        x = x.contiguous().view(x.size(0), 1, x.size(1), x.size(2))
        out = pad_layer_2d(x, self.in_conv_layer)
        out = self.act(out)
        out = self.norm_layer(out)
        for l in range(self.n_conv_blocks):
            y = pad_layer_2d(out, self.first_conv_layers[l])
            y = self.act(y)
            y = self.norm_layer(y)
            y = self.dropout_layer(y)
            y = pad_layer_2d(y, self.second_conv_layers[l])
            y = self.act(y)
            y = self.norm_layer(y)
            y = self.dropout_layer(y)
            if self.subsample[l] > 1:
                out = F.avg_pool2d(out, kernel_size=self.subsample[l], ceil_mode=True)
            out = y + out
        out = out.contiguous().view(out.size(0), out.size(1) * out.size(2), out.size(3))
        out = pad_layer(out, self.out_conv_layer)
        out = self.act(out)
        return out

class Postnet(nn.Module):
    def __init__(self, c_in, c_h, c_out, c_cond,  
            kernel_size, n_conv_blocks, 
            upsample, act, sn):
        super(Postnet, self).__init__()
        self.act = get_act(act)
        self.upsample = upsample
        self.c_h = c_h
        self.n_conv_blocks = n_conv_blocks
        f = spectral_norm if sn else lambda x: x
        total_upsample = reduce(lambda x, y: x*y, upsample)
        self.in_conv_layer = f(nn.Conv1d(c_in, c_h * c_out // total_upsample, kernel_size=1))
        self.first_conv_layers = nn.ModuleList([f(nn.Conv2d(c_h, c_h, kernel_size=kernel_size)) for _ \
                in range(n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList([f(nn.Conv2d(c_h, c_h*up*up, kernel_size=kernel_size)) 
            for up, _ in zip(upsample, range(n_conv_blocks))])
        self.out_conv_layer = f(nn.Conv2d(c_h, 1, kernel_size=1))
        self.conv_affine_layers = nn.ModuleList(
                [f(nn.Linear(c_cond, c_h * 2)) for _ in range(n_conv_blocks*2)])
        self.norm_layer = nn.InstanceNorm2d(c_h, affine=False)
        self.ps = nn.PixelShuffle(max(upsample))

    def forward(self, x, cond):
        out = pad_layer(x, self.in_conv_layer)
        out = out.contiguous().view(out.size(0), self.c_h, out.size(1) // self.c_h, out.size(2))
        for l in range(self.n_conv_blocks):
            y = pad_layer_2d(out, self.first_conv_layers[l])
            y = self.act(y)
            y = self.norm_layer(y)
            y = append_cond_2d(y, self.conv_affine_layers[l*2](cond))
            y = pad_layer_2d(y, self.second_conv_layers[l])
            y = self.act(y)
            if self.upsample[l] > 1:
                y = self.ps(y)
                y = self.norm_layer(y)
                y = append_cond_2d(y, self.conv_affine_layers[l*2+1](cond))
                out = y + upsample(out, scale_factor=(self.upsample[l], self.upsample[l])) 
            else:
                y = self.norm_layer(y)
                y = append_cond(y, self.conv_affine_layers[l*2+1](cond))
                out = y + out
        out = self.out_conv_layer(out)
        out = out.squeeze(dim=1)
        return out

class SpeakerEncoder(nn.Module):
    def __init__(self, c_in, c_h, c_out, kernel_size,
            bank_size, bank_scale, c_bank, 
            n_conv_blocks, n_dense_blocks, 
            subsample, act, dropout_rate):
        super(SpeakerEncoder, self).__init__()
        self.c_in = c_in
        self.c_h = c_h
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.n_conv_blocks = n_conv_blocks
        self.n_dense_blocks = n_dense_blocks
        self.subsample = subsample
        self.act = get_act(act)
        self.conv_bank = nn.ModuleList(
                [nn.Conv1d(c_in, c_bank, kernel_size=k) for k in range(bank_scale, bank_size + 1, bank_scale)])
        in_channels = c_bank * (bank_size // bank_scale) + c_in
        self.in_conv_layer = nn.Conv1d(in_channels, c_h, kernel_size=1)
        self.first_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size) for _ \
                in range(n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=sub) 
            for sub, _ in zip(subsample, range(n_conv_blocks))])
        self.pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.first_dense_layers = nn.ModuleList([nn.Linear(c_h, c_h) for _ in range(n_dense_blocks)])
        self.second_dense_layers = nn.ModuleList([nn.Linear(c_h, c_h) for _ in range(n_dense_blocks)])
        self.output_layer = nn.Linear(c_h, c_out)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def conv_blocks(self, inp):
        out = inp
        # convolution blocks
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])
            y = self.act(y)
            y = self.dropout_layer(y)
            y = pad_layer(y, self.second_conv_layers[l])
            y = self.act(y)
            y = self.dropout_layer(y)
            if self.subsample[l] > 1:
                out = F.avg_pool1d(out, kernel_size=self.subsample[l], ceil_mode=True)
            out = y + out
        return out

    def dense_blocks(self, inp):
        out = inp
        # dense layers
        for l in range(self.n_dense_blocks):
            y = self.first_dense_layers[l](out)
            y = self.act(y)
            y = self.dropout_layer(y)
            y = self.second_dense_layers[l](y)
            y = self.act(y)
            y = self.dropout_layer(y)
            out = y + out
        return out

    def forward(self, x):
        out = conv_bank(x, self.conv_bank, act=self.act)
        # dimension reduction layer
        out = pad_layer(out, self.in_conv_layer)
        out = self.act(out)
        # conv blocks
        out = self.conv_blocks(out)
        # avg pooling
        out = self.pooling_layer(out).squeeze(2)
        # dense blocks
        out = self.dense_blocks(out)
        out = self.output_layer(out)
        return out

class ContentEncoder(nn.Module):
    def __init__(self, c_in, c_h, c_out, kernel_size,
            bank_size, bank_scale, c_bank, 
            n_conv_blocks, subsample, 
            act, dropout_rate):
        super(ContentEncoder, self).__init__()
        self.n_conv_blocks = n_conv_blocks
        self.subsample = subsample
        self.act = get_act(act)
        self.conv_bank = nn.ModuleList(
                [nn.Conv1d(c_in, c_bank, kernel_size=k) for k in range(bank_scale, bank_size + 1, bank_scale)])
        in_channels = c_bank * (bank_size // bank_scale) + c_in
        self.in_conv_layer = nn.Conv1d(in_channels, c_h, kernel_size=1)
        self.first_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size) for _ \
                in range(n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=sub) 
            for sub, _ in zip(subsample, range(n_conv_blocks))])
        self.norm_layer = nn.InstanceNorm1d(c_h, affine=False)
        self.mean_layer = nn.Conv1d(c_h, c_out, kernel_size=1)
        self.std_layer = nn.Conv1d(c_h, c_out, kernel_size=1)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        out = conv_bank(x, self.conv_bank, act=self.act)
        # dimension reduction layer
        out = pad_layer(out, self.in_conv_layer)
        out = self.norm_layer(out)
        out = self.act(out)
        out = self.dropout_layer(out)
        # convolution blocks
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])
            y = self.norm_layer(y)
            y = self.act(y)
            y = self.dropout_layer(y)
            y = pad_layer(y, self.second_conv_layers[l])
            y = self.norm_layer(y)
            y = self.act(y)
            y = self.dropout_layer(y)
            if self.subsample[l] > 1:
                out = F.avg_pool1d(out, kernel_size=self.subsample[l], ceil_mode=True)
            out = y + out
        mu = pad_layer(out, self.mean_layer)
        log_sigma = pad_layer(out, self.std_layer)
        return mu, log_sigma

class Decoder(nn.Module):
    def __init__(self, 
            c_in, c_cond, c_h, c_out, 
            kernel_size,
            n_conv_blocks, upsample, act, sn, dropout_rate):
        super(Decoder, self).__init__()
        self.n_conv_blocks = n_conv_blocks
        self.upsample = upsample
        self.act = get_act(act)
        f = spectral_norm if sn else lambda x: x
        self.in_conv_layer = f(nn.Conv1d(c_in, c_h, kernel_size=1))
        self.first_conv_layers = nn.ModuleList([f(nn.Conv1d(c_h, c_h, kernel_size=kernel_size)) for _ \
                in range(n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList(\
                [f(nn.Conv1d(c_h, c_h * up, kernel_size=kernel_size)) \
                for _, up in zip(range(n_conv_blocks), self.upsample)])
        self.norm_layer = nn.InstanceNorm1d(c_h, affine=False)
        self.conv_affine_layers = nn.ModuleList(
                [f(nn.Linear(c_cond, c_h * 2)) for _ in range(n_conv_blocks*2)])
        self.out_conv_layer = f(nn.Conv1d(c_h, c_out, kernel_size=1))
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, z, cond):
        out = pad_layer(z, self.in_conv_layer)
        out = self.norm_layer(out)
        out = self.act(out)
        out = self.dropout_layer(out)
        # convolution blocks
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])
            y = self.norm_layer(y)
            y = append_cond(y, self.conv_affine_layers[l*2](cond))
            y = self.act(y)
            y = self.dropout_layer(y)
            y = pad_layer(y, self.second_conv_layers[l])
            if self.upsample[l] > 1:
                y = pixel_shuffle_1d(y, scale_factor=self.upsample[l])
            y = self.norm_layer(y)
            y = append_cond(y, self.conv_affine_layers[l*2+1](cond))
            y = self.act(y)
            y = self.dropout_layer(y)
            if self.upsample[l] > 1:
                out = y + upsample(out, scale_factor=self.upsample[l]) 
            else:
                out = y + out
        out = pad_layer(out, self.out_conv_layer)
        return out

class AE(nn.Module):
    def __init__(self, config):
        super(AE, self).__init__()
        self.speaker_encoder = SpeakerEncoder(**config['SpeakerEncoder']) 
        self.content_encoder = ContentEncoder(**config['ContentEncoder'])
        self.decoder = Decoder(**config['Decoder'])

    def forward(self, x):
        emb = self.speaker_encoder(x)
        mu, log_sigma = self.content_encoder(x)
        eps = log_sigma.new(*log_sigma.size()).normal_(0, 1)
        dec = self.decoder(mu + torch.exp(log_sigma / 2) * eps, emb)
        return mu, log_sigma, emb, dec

    def inference(self, x, x_cond):
        emb = self.speaker_encoder(x_cond)
        mu, _ = self.content_encoder(x)
        dec = self.decoder(mu, emb)
        return dec

    def get_speaker_embeddings(self, x):
        emb = self.speaker_encoder(x)
        return emb
