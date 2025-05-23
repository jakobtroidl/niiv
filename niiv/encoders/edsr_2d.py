# modified from: https://github.com/thstkdgus35/EDSR-PyTorch

import math
from argparse import Namespace

import torch
import torch.nn as nn
# import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm3d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm3d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class EDSR2D(nn.Module):
    def __init__(self, args=None, conv=default_conv):
        super(EDSR2D, self).__init__()
        if args is None:
            n_resblocks = 16
            n_feats = 64
            res_scale = 1
            n_colors = 1
        else:
            n_resblocks = args["n_resblocks"]
            n_feats = args["n_features"]
            n_colors = args["n_colors"]
            res_scale = args["res_scale"]
        kernel_size = 3
        act = nn.ReLU(True)

        # define head module
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        self.out_dim = n_feats

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = res
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


# def make_edsr_baseline(n_resblocks=16, n_feats=64, res_scale=1,
#                        scale=2, no_upsampling=False, rgb_range=1):
#     args = Namespace()
#     args.n_resblocks = n_resblocks
#     args.n_feats = n_feats
#     args.res_scale = res_scale

#     args.scale = [scale]
#     args.no_upsampling = no_upsampling

#     args.rgb_range = rgb_range
#     args.n_colors = 1
#     return EDSR(args)


# def make_edsr(n_resblocks=32, n_feats=256, res_scale=0.1,
#               scale=2, no_upsampling=False, rgb_range=1):
#     args = Namespace()
#     args.n_resblocks = n_resblocks
#     args.n_feats = n_feats
#     args.res_scale = res_scale

#     args.scale = [scale]
#     args.no_upsampling = no_upsampling

#     args.rgb_range = rgb_range
#     args.n_colors = 1
#     return EDSR(args)
