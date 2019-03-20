import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
import numpy as np
from math import ceil
from functools import reduce
from torch.nn.utils import spectral_norm
from utils import cc

class DummyStaticEncoder(object):
    def __init__(self, encoder):
        self.encoder = encoder

    def load(self, target_network):
        self.encoder.load_state_dict(target_network.state_dict())

    def __call__(self, x):
        return self.encoder(x)

def cal_gradpen(netD, real_data, real_cond, fake_data, fake_cond, center=0, alpha=None, device='cuda'):
    if alpha is not None:
        alpha = torch.tensor(alpha, device=device)  # torch.rand(real_data.size(0), 1, device=device)
    else:
        alpha = torch.rand(real_data.size(0), 1, device=device)
    alpha_exp = alpha.unsqueeze(2).expand(real_data.size())
    interpolates = alpha_exp * real_data + ((1 - alpha_exp) * fake_data)
    alpha_exp = alpha.expand(real_cond.size())
    interpolates_cond = alpha_exp * real_cond + ((1 - alpha_exp) * fake_cond) 
    interpolates.requires_grad_(True)
    interpolates_cond.requires_grad_(True)
    disc_interpolates = netD(interpolates, interpolates_cond)
    gradients_x = ag.grad(outputs=disc_interpolates, inputs=interpolates,
                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients_c = ag.grad(outputs=disc_interpolates, inputs=interpolates_cond,
                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty_x = ((gradients_x.norm(2, dim=1) - center) ** 2).mean()
    gradient_penalty_c = ((gradients_c.norm(2, dim=1) - center) ** 2).mean()
    return gradient_penalty_x + gradient_penalty_c

def compute_grad(d_out, x_in, center=0):
    # add activation sigmoid
    #d_out = torch.sigmoid(d_out)
    gradients = ag.grad(
            outputs=d_out, inputs=x_in, grad_outputs=d_out.new_ones(d_out.size()), 
            create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - center) ** 2).mean()
    return gradient_penalty

def pad_layer(inp, layer):
    kernel_size = layer.kernel_size[0]
    if kernel_size % 2 == 0:
        pad = (kernel_size//2, kernel_size//2 - 1)
    else:
        pad = (kernel_size//2, kernel_size//2)
    # padding
    inp = F.pad(inp, 
            pad=pad,
            mode='reflect')
    out = layer(inp)
    return out

def pad_layer_2d(inp, layer):
    kernel_size = layer.kernel_size[0]
    if kernel_size % 2 == 0:
        pad = (kernel_size//2, kernel_size//2 - 1, kernel_size//2, kernel_size//2 - 1)
    else:
        pad = (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2)
    # padding
    inp = F.pad(inp, 
            pad=pad,
            mode='reflect')
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

def conv_bank(x, module_list, act):
    outs = []
    for layer in module_list:
        out = act(pad_layer(x, layer))
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
    def __init__(self, c_in, c_h, n_blocks, act):
        super(MLP, self).__init__()
        self.act = get_act(act)
        self.n_blocks = n_blocks
        self.in_dense_layer = nn.Linear(c_in, c_h)
        self.first_dense_layers = nn.ModuleList([nn.Linear(c_h, c_h) for _ in range(n_blocks)])
        self.second_dense_layers = nn.ModuleList([nn.Linear(c_h, c_h) for _ in range(n_blocks)])

    def forward(self, x):
        h = self.in_dense_layer(x)
        for l in range(self.n_blocks):
            y = self.first_dense_layers[l](h)
            y = self.act(y)
            y = self.second_dense_layers[l](y)
            y = self.act(y)
            h = h + y
        return h

class StaticEncoder(nn.Module):
    def __init__(self, input_size, 
            c_in, c_h, c_out, kernel_size,
            bank_size, bank_scale, c_bank,
            n_conv_blocks, n_dense_blocks, 
            subsample, act, dropout_rate):
        super(StaticEncoder, self).__init__()
        self.input_size = input_size
        self.c_in = c_in
        self.c_h = c_h
        self.c_out = c_out
        self.c_bank = c_bank
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

class DynamicEncoder(nn.Module):
    def __init__(self, c_in, c_h, c_out, kernel_size, 
            bank_size, bank_scale, c_bank, 
            n_conv_blocks, subsample, n_dense_blocks, 
            act, dropout_rate):
        super(DynamicEncoder, self).__init__()

        self.c_in = c_in
        self.c_h = c_h
        self.c_out = c_out
        self.bank_size = bank_size
        self.bank_scale = bank_scale
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
        self.norm_layer = nn.InstanceNorm1d(c_h, affine=False)
        self.first_dense_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=1) \
                for _ in range(n_dense_blocks)])
        self.second_dense_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=1) \
                for _ in range(n_dense_blocks)])
        self.out_conv_layer = nn.Conv1d(c_h, c_out, kernel_size=1)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        out = conv_bank(x, self.conv_bank, act=self.act)
        # dimension reduction layer
        out = pad_layer(out, self.in_conv_layer)
        out = self.act(out)
        out = self.norm_layer(out)

        # convolution blocks
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])
            y = self.act(y)
            y = self.norm_layer(y)
            y = self.dropout_layer(y)
            y = pad_layer(y, self.second_conv_layers[l])
            y = self.act(y)
            y = self.norm_layer(y)
            y = self.dropout_layer(y)
            if self.subsample[l] > 1:
                out = F.avg_pool1d(out, kernel_size=self.subsample[l], ceil_mode=True)
            out = y + out

        for l in range(self.n_dense_blocks):
            y = self.first_dense_layers[l](out)
            y = self.act(y)
            y = self.norm_layer(y)
            y = self.dropout_layer(y)
            y = self.second_dense_layers[l](y)
            y = self.act(y)
            y = self.norm_layer(y)
            y = self.dropout_layer(y)
            out = y + out

        out = pad_layer(out, self.out_conv_layer)
        return out

# Conv_blocks followed by dense blocks
class Decoder(nn.Module):
    def __init__(self, c_in, c_cond, c_h, c_out, kernel_size, n_mlp_blocks,
            n_conv_blocks, upsample, n_dense_blocks, act, sn):
        super(Decoder, self).__init__()
        self.c_in = c_in
        self.c_h = c_h
        self.c_cond = c_cond
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.n_conv_blocks = n_conv_blocks
        self.n_dense_blocks = n_dense_blocks
        self.upsample = upsample
        self.act = get_act(act)
        self.mlp = MLP(c_in=c_cond, c_h=c_cond, n_blocks=n_mlp_blocks, act=act)
        f = spectral_norm if sn else lambda x: x
        self.in_conv_layer = f(nn.Conv1d(c_in, c_h, kernel_size=1))
        self.first_conv_layers = nn.ModuleList([f(nn.Conv1d(c_h, c_h, kernel_size=kernel_size)) for _ \
                in range(n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList(\
                [f(nn.Conv1d(c_h, c_h * up, kernel_size=kernel_size)) \
                for _, up in zip(range(n_conv_blocks), self.upsample)])
        self.norm_layer = nn.InstanceNorm1d(c_h, affine=False)
        self.conv_affine_layers = nn.ModuleList(
                [nn.Linear(c_cond, c_h * 2) for _ in range(n_conv_blocks*2)])
        self.first_dense_layers = nn.ModuleList([f(nn.Conv1d(c_h, c_h, kernel_size=1)) \
                for _ in range(n_dense_blocks)])
        self.second_dense_layers = nn.ModuleList([f(nn.Conv1d(c_h, c_h, kernel_size=1)) \
                for _ in range(n_dense_blocks)])
        self.dense_affine_layers = nn.ModuleList(
                [nn.Linear(c_cond, c_h * 2) for _ in range(n_dense_blocks*2)])
        self.out_conv_layer = f(nn.Conv1d(c_h, c_out, kernel_size=1))

    def forward(self, x, cond):
        out = pad_layer(x, self.in_conv_layer)
        out = self.act(out)
        cond = self.mlp(cond)
        # convolution blocks
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])
            y = self.act(y)
            y = self.norm_layer(y)
            y = append_cond(y, self.conv_affine_layers[l*2](cond))
            y = pad_layer(y, self.second_conv_layers[l])
            y = self.act(y)
            if self.upsample[l] > 1:
                y = pixel_shuffle_1d(y, scale_factor=self.upsample[l])
                y = self.norm_layer(y)
                y = append_cond(y, self.conv_affine_layers[l*2+1](cond))
                out = y + upsample(out, scale_factor=self.upsample[l]) 
            else:
                y = self.norm_layer(y)
                y = append_cond(y, self.conv_affine_layers[l*2+1](cond))
                out = y + out

        for l in range(self.n_dense_blocks):
            y = self.first_dense_layers[l](y)
            y = self.act(y)
            y = self.norm_layer(y)
            y = append_cond(y, self.dense_affine_layers[l*2](cond))
            y = self.second_dense_layers[l](y)
            y = self.act(y)
            y = self.norm_layer(y)
            y = append_cond(y, self.dense_affine_layers[l*2+1](cond))
            out = y + out
        out = pad_layer(out, self.out_conv_layer)
        return out

class AE(nn.Module):
    def __init__(self, input_size, 
            c_in, s_c_h, d_c_h,
            c_latent, c_cond,
            c_bank, bank_size, bank_scale,
            c_out, kernel_size,
            s_enc_n_conv_blocks, s_enc_n_dense_blocks,
            d_enc_n_conv_blocks, d_enc_n_dense_blocks,
            s_subsample, d_subsample, 
            dec_n_conv_blocks, dec_n_dense_blocks,
            dec_n_mlp_blocks,
            upsample, act, dropout_rate, use_dummy, sn):
        super(AE, self).__init__()
        self.use_dummy = use_dummy
        self.static_encoder = StaticEncoder(input_size=input_size, 
                c_in=c_in, c_h=s_c_h, c_out=c_cond, 
                c_bank=c_bank,
                bank_size=bank_size, bank_scale=bank_scale,
                kernel_size=kernel_size, 
                n_conv_blocks=s_enc_n_conv_blocks, 
                subsample=s_subsample,
                n_dense_blocks=s_enc_n_dense_blocks, 
                act=act, dropout_rate=dropout_rate)
        if use_dummy:
            # dummy system
            self.dummy_static_encoder = DummyStaticEncoder(cc(StaticEncoder(input_size=input_size, 
                    c_in=c_in, c_h=s_c_h, c_out=c_cond, 
                    c_bank=c_bank,
                    bank_size=bank_size, bank_scale=bank_scale,
                    kernel_size=kernel_size, 
                    n_conv_blocks=s_enc_n_conv_blocks, 
                    subsample=s_subsample,
                    n_dense_blocks=s_enc_n_dense_blocks, 
                    act=act, dropout_rate=dropout_rate)))

        self.dynamic_encoder = DynamicEncoder(c_in=c_in, c_h=d_c_h, c_out=c_latent, 
                c_bank=c_bank,
                bank_size=bank_size, bank_scale=bank_scale,
                kernel_size=kernel_size, 
                n_conv_blocks=d_enc_n_conv_blocks, 
                subsample=d_subsample, 
                n_dense_blocks=d_enc_n_dense_blocks, 
                act=act, dropout_rate=dropout_rate)

        self.decoder = Decoder(c_in=c_latent, c_cond=c_cond, 
                c_h=d_c_h, c_out=c_out, 
                kernel_size=kernel_size,
                n_mlp_blocks=dec_n_mlp_blocks,
                n_conv_blocks=dec_n_conv_blocks, 
                upsample=upsample, 
                n_dense_blocks=dec_n_dense_blocks, 
                act=act, sn=sn)

    def forward(self, x, x_pos, x_neg, mode):
        # for autoencoder pretraining
        if mode == 'pretrain_ae': 
            # static operation
            emb_pos = self.static_encoder(x_pos)
            # dynamic operation
            enc = self.dynamic_encoder(x)
            # decode
            d_noise = enc.new(*enc.size()).normal_(0, 1)
            dec = self.decoder(enc + d_noise, emb_pos)
            return enc, emb_pos, dec
        elif mode == 'gen_ae':
            with torch.no_grad():
                # static operation
                emb_neg = self.static_encoder(x_neg)
                # dynamic operation
                enc = self.dynamic_encoder(x)
            # synthesis with emb_neg 
            d_noise = enc.new(*enc.size()).normal_(0, 1)
            dec_syn = self.decoder(enc.detach() + d_noise, emb_neg.detach())
            # rec emb, using dummy encoder to avoid grad update
            if self.use_dummy:
                self.dummy_static_encoder.load(self.static_encoder)
                emb_rec = self.dummy_static_encoder(dec_syn)
            else:
                emb_rec = self.static_encoder(dec_syn)
            return enc, emb_neg, emb_rec, dec_syn
        elif mode == 'dis_real':
            emb = self.static_encoder(x)
            return emb
        elif mode == 'dis_fake':
            # dynamic operation
            enc = self.dynamic_encoder(x)
            emb_neg = self.static_encoder(x_neg)
            d_noise = enc.new(*enc.size()).normal_(0, 1)
            dec_syn = self.decoder(enc + d_noise, emb_neg)
            return enc, emb_neg, dec_syn
        elif mode == 'dis_mismatch':
            emb_neg = self.static_encoder(x_neg)
            return emb_neg

    def inference(self, x, x_cond):
        emb = self.static_encoder(x_cond)
        enc = self.dynamic_encoder(x)
        dec = self.decoder(enc, emb)
        return dec

    def get_static_embeddings(self, x):
        out = self.static_encoder(x)
        return out

class Discriminator(nn.Module):
    def __init__(self, input_size, 
            c_in, c_h, c_cond, 
            kernel_size, n_conv_blocks,
            subsample, 
            n_dense_layers, d_h, act, sn):
        super(Discriminator, self).__init__()
        # input_size is a tuple
        self.n_conv_blocks = n_conv_blocks
        self.n_dense_layers = n_dense_layers
        self.subsample = subsample
        self.act = get_act(act)
        # using spectral_norm if specified, or identity function
        f = spectral_norm if sn else lambda x: x
        self.in_conv_layer = f(nn.Conv2d(c_in, c_h, kernel_size=kernel_size))
        self.first_conv_layers = nn.ModuleList(
                [f(nn.Conv2d(c_h, c_h, kernel_size=kernel_size)) for _ in range(self.n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList(
                [f(nn.Conv2d(c_h, c_h, kernel_size=kernel_size, stride=(2, sub))) for sub in subsample])
        # to process all frequency
        dense_input_size = input_size 
        for l, sub in zip(range(n_conv_blocks), self.subsample):
            dense_input_size = (ceil(dense_input_size[0] / 2), ceil(dense_input_size[1] / sub))
        self.out_conv_layer = f(nn.Conv2d(c_h, d_h, \
                kernel_size=(dense_input_size[0], kernel_size), \
                stride=(1, 1), padding=(0, kernel_size // 2)))
        self.combine_layer = f(nn.Conv1d(d_h + c_cond, d_h, kernel_size=1))
        dense_input_size = dense_input_size[1] * d_h
        self.dense_layers = nn.ModuleList([f(nn.Linear(dense_input_size + c_cond, d_h))] + 
                [f(nn.Linear(d_h + c_cond, d_h)) for _ in range(n_dense_layers - 2)] + 
                [f(nn.Linear(d_h + c_cond, 1))])

    def conv_blocks(self, inp, cond):
        out = self.act(pad_layer_2d(inp, self.in_conv_layer))
        for l in range(self.n_conv_blocks):
            y = self.act(pad_layer_2d(out, self.first_conv_layers[l]))
            y = self.act(pad_layer_2d(y, self.second_conv_layers[l]))
            out = y + F.avg_pool2d(out, kernel_size=(2, self.subsample[l]), ceil_mode=True)
        out = self.out_conv_layer(out).squeeze(2)
        out = self.act(out)
        out = self.combine_layer(concat_cond(out, cond))
        out = self.act(out)
        out = out.view(out.size(0), out.size(1) * out.size(2))
        return out

    def dense_blocks(self, inp, cond):
        h = inp
        for l in range(self.n_dense_layers - 1):
            h = torch.cat([h, cond], dim=1)
            h = self.dense_layers[l](h)
            h = self.act(h)
        h = torch.cat([h, cond], dim=1)
        out = self.dense_layers[-1](h)
        return out

    def forward(self, x, cond):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x_vec = self.conv_blocks(x, cond)
        out = self.dense_blocks(x_vec, cond)
        return out

