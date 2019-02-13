import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import ceil

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

def pixel_shuffle_1d(inp, scale_factor=2):
    batch_size, channels, in_width = inp.size()
    channels //= scale_factor
    out_width = in_width * scale_factor
    inp_view = inp.contiguous().view(batch_size, channels, scale_factor, in_width)
    shuffle_out = inp_view.permute(0, 1, 3, 2).contiguous()
    shuffle_out = shuffle_out.view(batch_size, channels, out_width)
    return shuffle_out

def upsample(x, scale_factor=2):
    x_up = F.interpolate(x, scale_factor=2, mode='nearest')
    return x_up

def append_cond(x, cond):
    # x = [batch_size, x_channels, length]
    # cond = [batch_size, x_channels]
    cond = cond.unsqueeze(dim=2)
    cond = cond.expand(*cond.size()[:-1], x.size(-1))
    out = torch.cat([x, cond], dim=1)
    #out = x + cond
    return out

def conv_bank(x, module_list, act):
    outs = []
    for layer in module_list:
        out = pad_layer(x, layer)
        outs.append(out)
    outs = torch.cat(outs, dim=1)
    outs = act(outs)
    out = torch.cat([outs, x], dim=1)
    return out

def get_act(act):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'lrelu':
        return nn.LeakyReLU()
    else:
        return nn.ReLU()

# Conv_blocks followed by dense blocks
class Encoder(nn.Module):
    def __init__(self, c_in, c_h, c_out, kernel_size, bank_size, bank_scale, 
            n_conv_blocks, subsample, n_dense_blocks, act, dropout_rate):
        super(Encoder, self).__init__()

        self.c_in = c_in
        self.c_h = c_h
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.bank_size = bank_size
        self.bank_scale = bank_scale
        self.n_conv_blocks = n_conv_blocks
        self.n_dense_blocks = n_dense_blocks
        self.subsample = subsample
        self.act = get_act(act)
        self.conv_bank = nn.ModuleList(
                [nn.Conv1d(c_in, c_h, kernel_size=k) for k in \
                range(bank_scale, bank_size + 1, bank_scale)])
        in_channels = c_h * (bank_size // bank_scale) + c_in
        self.in_conv_layer = nn.Conv1d(in_channels, c_h, kernel_size=1)
        self.first_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size) for _ \
                in range(n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=sub) 
            for sub, _ in zip(subsample, range(n_conv_blocks))])
        self.conv_norm_layers = nn.ModuleList([nn.InstanceNorm1d(c_h) for _ in range(n_conv_blocks)])
        self.first_dense_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=1) \
                for _ in range(n_dense_blocks)])
        self.second_dense_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=1) \
                for _ in range(n_dense_blocks)])
        self.dense_norm_layers = nn.ModuleList([nn.InstanceNorm1d(c_h) for _ in range(n_dense_blocks)])
        self.out_conv_layer = nn.Conv1d(c_h, c_out, kernel_size=1)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        out = conv_bank(x, self.conv_bank, act=self.act)
        # dimension reduction layer
        out = pad_layer(out, self.in_conv_layer)

        # convolution blocks
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])
            y = self.act(y)
            y = self.dropout_layer(y)
            y = pad_layer(y, self.second_conv_layers[l])
            y = self.act(y)
            y = self.conv_norm_layers[l](y)
            y = self.dropout_layer(y)
            if self.subsample[l] > 1:
                out = F.avg_pool1d(out, kernel_size=self.subsample[l])
            out = y + out

        for l in range(self.n_dense_blocks):
            y = self.first_dense_layers[l](out)
            y = self.act(y)
            y = self.dropout_layer(y)
            y = self.second_dense_layers[l](y)
            y = self.act(y)
            y = self.dense_norm_layers[l](y)
            y = self.dropout_layer(y)
            out = out + y

        out = pad_layer(out, self.out_conv_layer)
        return out

# Conv_blocks followed by dense blocks
class Decoder(nn.Module):
    def __init__(self, c_in, c_cond, c_h, c_out, kernel_size, 
            n_conv_blocks, upsample, n_dense_blocks, act, dropout_rate):
        super(Decoder, self).__init__()
        self.c_in = c_in
        self.c_h = c_h
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.n_conv_blocks = n_conv_blocks
        self.n_dense_blocks = n_dense_blocks
        self.upsample = upsample
        self.act = get_act(act)
        self.in_conv_layer = nn.Conv1d(c_in, c_h, kernel_size=1)
        self.first_conv_layers = nn.ModuleList([nn.Conv1d(c_h + c_cond, c_h, kernel_size=kernel_size) for _ \
                in range(n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList(\
                [nn.Conv1d(c_h + c_cond, c_h * up, kernel_size=kernel_size) \
                for _, up in zip(range(n_conv_blocks), self.upsample)])
        self.conv_norm_layers = nn.ModuleList(\
                [nn.InstanceNorm1d(c_h * up) for _, up in zip(range(n_conv_blocks), self.upsample)])
        self.first_dense_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=1) \
                for _ in range(n_dense_blocks)])
        self.second_dense_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=1) \
                for _ in range(n_dense_blocks)])
        self.dense_norm_layers = nn.ModuleList([nn.InstanceNorm1d(c_h) for _ in range(n_dense_blocks)])
        self.out_conv_layer = nn.Conv1d(c_h, c_out, kernel_size=1)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, x, cond):
        out = pad_layer(x, self.in_conv_layer)
        cond = self.cond_linear_layer(cond)
        # convolution blocks
        for l in range(self.n_conv_blocks):
            y = append_cond(out, cond)
            y = pad_layer(y, self.first_conv_layers[l])
            y = self.act(y)
            y = self.dropout_layer(y)
            y = append_cond(y, cond)
            y = pad_layer(y, self.second_conv_layers[l])
            y = self.act(y)
            if self.upsample[l] > 1:
                y = pixel_shuffle_1d(y, scale_factor=self.upsample[l])
                y = self.conv_norm_layers[l](y)
                y = self.dropout_layer(y)
                out = y + upsample(out, scale_factor=self.upsample[l]) 
            else:
                y = self.conv_norm_layers[l](y)
                y = self.dropout_layer(y)
                out = y + out

        for l in range(self.n_dense_blocks):
            y = append_cond(out, cond)
            y = self.first_dense_layers[l](y)
            y = self.act(y)
            y = self.dropout_layer(y)
            y = append_cond(y, cond)
            y = self.second_dense_layers[l](y)
            y = self.act(y)
            y = self.dense_norm_layers[l](y)
            y = self.dropout_layer(y)
            out = out + y
        out = pad_layer(out, self.out_conv_layer)
        return out

class AE(nn.Module):
    def __init__(self, c_in, c_h,
            c_latent, c_cond,
            c_out, kernel_size,
            bank_size, bank_scale,
            s_enc_n_conv_blocks, s_enc_n_dense_blocks, 
            d_enc_n_conv_blocks, d_enc_n_dense_blocks,
            s_subsample, d_subsample, 
            dec_n_conv_blocks, dec_n_dense_blocks, upsample, act, dropout_rate):
        super(AE, self).__init__()
        self.static_encoder = Encoder(c_in=c_in, c_h=c_h, c_out=c_cond, 
                kernel_size=kernel_size, 
                bank_size=bank_size, bank_scale=bank_scale,
                n_conv_blocks=s_enc_n_conv_blocks, 
                subsample=s_subsample, 
                n_dense_blocks=s_enc_n_dense_blocks, 
                act=act, dropout_rate=dropout_rate)
        self.dynamic_encoder = Encoder(c_in=c_in, c_h=c_h, c_out=c_latent, 
                kernel_size=kernel_size, 
                bank_size=bank_size, bank_scale=bank_scale,
                n_conv_blocks=d_enc_n_conv_blocks, 
                subsample=d_subsample, 
                n_dense_blocks=d_enc_n_dense_blocks, 
                act=act, dropout_rate=dropout_rate)
        self.decoder = Decoder(c_in=c_latent, c_cond=c_cond, 
                c_h=c_h, c_out=c_out, 
                kernel_size=kernel_size, 
                n_conv_blocks=dec_n_conv_blocks, 
                upsample=upsample, 
                n_dense_blocks=dec_n_dense_blocks, 
                act=act, dropout_rate=dropout_rate)

    def static_operation(self, x):
        enc = self.static_encoder(x)
        emb = F.avg_pool1d(enc, kernel_size=enc.size(2)).squeeze(2)
        return emb 

    def forward(self, x, x_pos, x_neg, mode):
        # for autoencoder pretraining
        if mode == 'ae': 
            # static operation
            emb_pos = self.static_operation(x_pos)
            # dynamic operation
            enc = self.dynamic_encoder(x)
            # decode
            dec = self.decoder(enc, emb_pos)
            return enc, emb_pos, dec
        elif mode == 'latent_ae':
            # static operation
            emb = self.static_operation(x)
            emb_pos = self.static_operation(x_pos)
            # dynamic operation
            enc = self.dynamic_encoder(x)
            enc_pos = self.dynamic_encoder(x_pos)
            enc_neg = self.dynamic_encoder(x_neg)
            # decode
            dec = self.decoder(enc, emb_pos)
            return enc, enc_pos, enc_neg, emb, emb_pos, dec
        elif mode == 'latent_dis':
            # dynamic operation
            enc = self.dynamic_encoder(x)
            enc_pos = self.dynamic_encoder(x_pos)
            enc_neg = self.dynamic_encoder(x_neg)
            return enc, enc_pos, enc_neg 
        elif mode == 'raw_ae':
            with torch.no_grad():
                # static operation
                emb_pos = self.static_operation(x_pos)
                emb_neg = self.static_operation(x_neg)
                # dynamic operation
                enc = self.dynamic_encoder(x)
                enc_pos = self.dynamic_encoder(x_pos)
            # decode
            dec = self.decoder(enc, emb_pos)
            dec_syn = self.decoder(enc_pos, emb_neg)
            return enc, enc_pos, emb_pos, emb_neg, dec, dec_syn
        elif mode == 'raw_dis':
            # static operation
            emb_neg = self.static_operation(x_neg)
            # dynamic operation
            enc = self.dynamic_encoder(x)
            dec_syn = self.decoder(enc, emb_neg)
            return enc, emb_neg, dec_syn

class LatentDiscriminator(nn.Module):
    def __init__(self, input_size, c_in, c_h, kernel_size, 
            n_conv_layers, n_dense_layers, d_h, act, dropout_rate):
        super(LatentDiscriminator, self).__init__()
        self.input_size = input_size
        self.c_in = c_in
        self.c_h = c_h
        self.kernel_size = kernel_size
        self.n_conv_layers = n_conv_layers
        self.n_dense_layers = n_dense_layers
        self.d_h = d_h
        self.act = get_act(act)
        self.in_conv_layer = nn.Conv1d(c_in, c_h, kernel_size=kernel_size)
        self.conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=2) \
                for _ in range(n_conv_layers)])
        self.dense_layers = nn.ModuleList([nn.Linear(c_h * 2, d_h)] + 
                [nn.Linear(d_h, d_h) for _ in range(n_dense_layers - 2)] + 
                [nn.Linear(d_h, 1)])
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def conv_blocks(self, inp):
        out = pad_layer(inp, self.in_conv_layer)
        for l in range(self.n_conv_layers):
            out = pad_layer(out, self.conv_layers[l])
            out = self.act(out)
            out = self.dropout_layer(out)
        out = F.avg_pool1d(out, kernel_size=out.size(2))
        out = out.squeeze(dim=2)
        return out

    def dense_blocks(self, inp):
        out = inp
        for l in range(self.n_dense_layers - 1):
            out = self.dense_layers[l](out)
            out = self.act(out)
            out = self.dropout_layer(out)
        out = self.dense_layers[-1](out)
        return out

    def forward(self, x, x_pos, x_neg):
        x_vec = self.conv_blocks(x)
        x_pos_vec = self.conv_blocks(x_pos)
        x_neg_vec = self.conv_blocks(x_neg)

        fused_pos = torch.cat([x_vec, x_pos_vec], dim=1)
        fused_neg = torch.cat([x_vec, x_neg_vec], dim=1)

        pos_val = self.dense_blocks(fused_pos)
        neg_val = self.dense_blocks(fused_neg)
        return pos_val, neg_val

if __name__ == '__main__':
    ae = AE(c_in=1, c_h=64, c_out=1, c_cond=32, 
            kernel_size=60, 
            bank_size=150, bank_scale=20, 
            s_enc_n_conv_blocks=3, 
            s_enc_n_dense_blocks=2, 
            d_enc_n_conv_blocks=5, 
            d_enc_n_dense_blocks=3, 
            s_subsample=[2, 2, 2], 
            d_subsample=[1, 2, 2, 2, 1], 
            dec_n_conv_blocks=5, 
            dec_n_dense_blocks=2, 
            upsample=[1, 1, 2, 2, 2], 
            act='lrelu', dropout_rate=0.5).cuda()
    print(ae)
    D = LatentDiscriminator(input_size=1000, c_in=128, c_h=256, kernel_size=60, 
            n_conv_layers=4, d_h=512, act='lrelu', dropout_rate=0.5).cuda()
    data = torch.randn(5, 1, 2000, device='cuda')
    data_pos = torch.randn(5, 1, 8000, device='cuda')
    data_neg = torch.randn(5, 1, 8000, device='cuda')
    enc, enc_pos, enc_neg, dec, emb, emb_pos = ae(data, data_pos, data_neg)
    o = D(torch.cat([enc, enc_pos], dim=1))
