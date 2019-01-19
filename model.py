import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
    x_up = F.upsample(x, scale_factor=2, mode='nearest')
    return x_up

def append_cond(x, cond):
    # x = [batch_size, x_channels, length]
    # cond = [batch_size, cond_channels]
    cond = cond.unsqueeze(dim=2)
    cond_exp = cond.expand(cond.size(0), cond.size(1), x.size(2))
    out = torch.cat([x, cond_exp], dim=1)
    return out

# Conv_blocks followed by dense blocks
class Encoder(nn.Module):
    def __init__(self, c_in, c_h, c_out, kernel_size, n_conv_blocks, subsample, n_dense_blocks, act):
        super(Encoder, self).__init__()

        self.c_in = c_in
        self.c_h = c_h
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.n_conv_blocks = n_conv_blocks
        self.n_dense_blocks = n_dense_blocks
        self.subsample = subsample
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'lrelu':
            self.act = nn.LeakyReLU()
        else:
            self.act = nn.ReLU()
        self.in_conv_layer = nn.Conv1d(c_in, c_h, kernel_size=kernel_size)
        self.first_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size) for _ \
                in range(n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size) for _ \
                in range(n_conv_blocks)])
        self.conv_norm_layers = nn.ModuleList([nn.InstanceNorm1d(c_h) for _ in range(n_conv_blocks)])
        self.first_dense_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=1) \
                for _ in range(n_dense_blocks)])
        self.second_dense_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=1) \
                for _ in range(n_dense_blocks)])
        self.dense_norm_layers = nn.ModuleList([nn.InstanceNorm1d(c_h) for _ in range(n_dense_blocks)])
        self.out_conv_layer = nn.Conv1d(c_h, c_out, kernel_size=1)

    def forward(self, x):
        # first convolution layer
        out = pad_layer(x, self.in_conv_layer)

        # convolution blocks
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])
            y = self.act(y)
            y = pad_layer(y, self.second_conv_layers[l])
            y = self.act(y)
            y = self.conv_norm_layers[l](y)
            out = y + out
            if self.subsample[l] > 1:
                out = F.avg_pool1d(out, kernel_size=self.subsample[l])

        for l in range(self.n_dense_blocks):
            y = self.first_dense_layers[l](out)
            y = self.act(y)
            y = self.second_dense_layers[l](y)
            y = self.act(y)
            y = self.dense_norm_layers[l](y)
            out = out + y

        out = pad_layer(out, self.out_conv_layer)
        return out

# Conv_blocks followed by dense blocks
class Decoder(nn.Module):
    def __init__(self, c_in, c_h, c_out, c_cond, kernel_size, n_conv_blocks, upsample, n_dense_blocks, act):
        super(Decoder, self).__init__()
        self.c_in = c_in
        self.c_h = c_h
        self.c_out = c_out
        self.c_cond = c_cond
        self.kernel_size = kernel_size
        self.n_conv_blocks = n_conv_blocks
        self.n_dense_blocks = n_dense_blocks
        self.upsample = upsample
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'lrelu':
            self.act = nn.LeakyReLU()
        else:
            self.act = nn.ReLU()
        self.in_conv_layer = nn.Conv1d(c_in, c_h, kernel_size=kernel_size)
        self.first_conv_layers = nn.ModuleList([nn.Conv1d(c_h + c_cond, c_h, kernel_size=kernel_size) for _ \
                in range(n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList(\
                [nn.Conv1d(c_h + c_cond, c_h * up, kernel_size=kernel_size) \
                for _, up in zip(range(n_conv_blocks), self.upsample)])
        self.conv_norm_layers = nn.ModuleList(\
                [nn.InstanceNorm1d(c_h * up) for _, up in zip(range(n_conv_blocks), self.upsample)])
        self.first_dense_layers = nn.ModuleList([nn.Conv1d(c_h + c_cond, c_h, kernel_size=1) \
                for _ in range(n_dense_blocks)])
        self.second_dense_layers = nn.ModuleList([nn.Conv1d(c_h + c_cond, c_h, kernel_size=1) \
                for _ in range(n_dense_blocks)])
        self.dense_norm_layers = nn.ModuleList([nn.InstanceNorm1d(c_h) for _ in range(n_dense_blocks)])
        self.out_conv_layer = nn.Conv1d(c_h, c_out, kernel_size=1)

    def forward(self, x, cond):
        out = pad_layer(x, self.in_conv_layer)
        # convolution blocks
        for l in range(self.n_conv_blocks):
            y = append_cond(out, cond)
            y = pad_layer(y, self.first_conv_layers[l])
            y = self.act(y)
            y = append_cond(y, cond)
            y = pad_layer(y, self.second_conv_layers[l])
            y = self.act(y)
            if self.upsample[l] > 1:
                y = pixel_shuffle_1d(y, scale_factor=self.upsample[l])
                y = self.conv_norm_layers[l](y)
                out = y + upsample(out, scale_factor=self.upsample[l]) 
            else:
                y = self.conv_norm_layers[l](y)
                out = y + out

        for l in range(self.n_dense_blocks):
            y = append_cond(out, cond)
            y = self.first_dense_layers[l](y)
            y = self.act(y)
            y = append_cond(y, cond)
            y = self.second_dense_layers[l](y)
            y = self.act(y)
            y = self.dense_norm_layers[l](y)
            out = out + y
        out = pad_layer(out, self.out_conv_layer)
        return out

class AE(nn.Module):
    def __init__(self, c_in, c_h, c_out, c_cond, kernel_size, 
            s_enc_n_conv_blocks, s_enc_n_dense_blocks, 
            d_enc_n_conv_blocks, d_enc_n_dense_blocks,
            s_subsample, d_subsample,  
            dec_n_conv_blocks, dec_n_dense_blocks, upsample, act):
        super(AE, self).__init__()
        self.static_encoder = Encoder(c_in=c_in, c_h=c_h, c_out=c_cond, 
                kernel_size=kernel_size, 
                n_conv_blocks=s_enc_n_conv_blocks, 
                subsample=s_subsample, 
                n_dense_blocks=s_enc_n_dense_blocks, 
                act=act)
        self.dynamic_encoder = Encoder(c_in=c_in, c_h=c_h, c_out=c_h, 
                kernel_size=kernel_size, 
                n_conv_blocks=d_enc_n_conv_blocks, 
                subsample=d_subsample, 
                n_dense_blocks=d_enc_n_dense_blocks, 
                act=act)
        self.decoder = Decoder(c_in=c_h, c_h=c_h, c_out=c_out, c_cond=c_cond, 
                kernel_size=kernel_size, 
                n_conv_blocks=dec_n_conv_blocks, 
                upsample=upsample, 
                n_dense_blocks=dec_n_dense_blocks, 
                act=act)

    def static_operation(self, x):
        enc = self.static_encoder(x)
        emb = F.avg_pool1d(enc, kernel_size=enc.size(2)).squeeze(2)
        return emb 

    def forward(self, x, x_pos, x_neg):
        # static operation
        emb = self.static_operation(x)
        emb_pos = self.static_operation(x_pos)

        # dynamic operation
        enc = self.dynamic_encoder(x)
        enc_pos = self.dynamic_encoder(x_pos)
        enc_neg = self.dynamic_encoder(x_neg)

        # decode
        dec = self.decoder(enc, emb_pos)

        return enc, enc_pos, enc_neg, dec, emb, emb_pos

if __name__ == '__main__':
    ae = AE(c_in=1, c_h=32, c_out=1, c_cond=32, 
            kernel_size=60, 
            s_enc_n_conv_blocks=3, 
            s_enc_n_dense_blocks=2, 
            d_enc_n_conv_blocks=5, 
            d_enc_n_dense_blocks=3, 
            s_subsample=[2, 2, 2], 
            d_subsample=[1, 2, 2, 2, 1], 
            dec_n_conv_blocks=5, 
            dec_n_dense_blocks=2, 
            upsample=[1, 1, 2, 2, 2], 
            act='lrelu').cuda()
    data = torch.randn(5, 1, 8000, device='cuda')
    data_pos = torch.randn(5, 1, 8000, device='cuda')
    data_neg = torch.randn(5, 1, 8000, device='cuda')
    all_items = ae(data, data_pos, data_neg)
    for a in all_items:
        print(a.size())
