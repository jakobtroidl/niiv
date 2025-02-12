import torch
import math
import numpy as np

from torch import nn
from niiv.feature_grid import FeatureGrid
from niiv.encoders.edsr_2d import EDSR2D
from niiv.decoder.mlp import MLP
from niiv.encoders import rdn
from niiv.encoders import swinir
from niiv.decoder import inr
from niiv.decoder.field_siren import FieldSiren
from timm.models.layers import DropPath
from linformer import LinformerSelfAttention
from PIL import Image


from torch import nn, einsum
from einops import rearrange, repeat

class NIIV(nn.Module):
    def __init__(self, out_features=1, encoding_config=None, n_pos_enc_octaves=2, **kwargs):
        super().__init__()

        self.feat_unfold = False

        if encoding_config is None:
            n_features = 64
            n_layers = 5
            n_neurons = 256
            self.encoder = EDSR2D()
        else:
            # hyper parameters
            n_features = encoding_config["encoder"]["n_features"]
            n_layers = encoding_config["network"]["n_hidden_layers"]
            n_neurons = encoding_config["network"]["n_neurons"]
            self.encoder = EDSR2D(args = encoding_config["encoder"])   

        # module for latent grid processing
        self.grid = FeatureGrid(feat_unfold=self.feat_unfold, n_pos_encoding=n_pos_enc_octaves)
        model_in = self.grid.n_out(n_features) 


        

        # trainable parameters
        # self.encoder = rdn.make_rdn()
        # self.encoder = swinir.make_swinir()
        self.decoder = MLP(in_dim=model_in, out_dim=out_features, n_neurons=n_neurons, n_hidden=n_layers)

    def forward(self, image, coords):
        latent_grid = self.encoder(image)
        features = self.grid.compute_features(image, latent_grid, coords)
        bs, q = coords.squeeze(1).squeeze(1).shape[:2]
        prediction = self.decoder(features.view(bs * q, -1)).view(bs, q, -1)
        return torch.clamp(prediction, 0, 1)

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, drop_path_rate = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.drop_path(self.to_out(out))



class NIIV_Attn(nn.Module):
    def __init__(self, out_features=1, encoding_config=None, n_pos_enc_octaves=2, **kwargs):
        super().__init__()

        self.feat_unfold = False

        if encoding_config is None:
            n_features = 64
            n_layers = 5
            n_neurons = 256
            self.encoder = EDSR2D()
        else:
            # hyper parameters
            n_features = encoding_config["encoder"]["n_features"]
            n_layers = encoding_config["network"]["n_hidden_layers"]
            n_neurons = encoding_config["network"]["n_neurons"]
            self.encoder = EDSR2D(args = encoding_config["encoder"])   

        # module for latent grid processing
        self.grid = FeatureGrid(feat_unfold=self.feat_unfold, n_pos_encoding=n_pos_enc_octaves)
        model_in = self.grid.n_out(n_features)

        self.mode = encoding_config["mode"]

        # self.use_attn = encoding_config["cross_attn"]
        # self.use_concat = encoding_config["feat_concat"]
        # trainable parameters
        # self.encoder = rdn.make_rdn()
        # self.encoder = swinir.make_swinir()

        if self.mode == "feat_concat": # concat features for each degradation direction
            model_in = model_in * 2

        self.cross_attn = PreNorm(model_in, Attention(model_in, model_in, heads = 1, dim_head = model_in), context_dim = model_in)
        self.lin_attn = LinformerSelfAttention(dim = model_in, seq_len = 16384, heads = 1, k = 32, one_kv_head = True, share_kv = True)

        self.decoder = MLP(in_dim=model_in, out_dim=out_features, n_neurons=n_neurons, n_hidden=n_layers)
    

    def encode(self, input, coords):
        latent_grid = self.encoder(input)
        features = self.grid.compute_features(input, latent_grid, coords)
        return features
    
    def decode(self, xz_features, yz_features):
        # rotate yz_features to align with xz_features
        bs, q = xz_features.shape[:2]
        im_size = int(math.sqrt(q))

        # view tensors volumetrically
        xz_features = xz_features.view(bs, im_size, im_size, -1) # yxz
        yz_features = yz_features.view(bs, im_size, im_size, -1) # xyz

        # realign both feature grid to xyz
        xz_features = xz_features.permute(1, 0, 2, 3) # xyz
        # yz_features = yz_features.permute(2, 0, 1, 3) # xyz

        # concatenate the features
        if self.mode == "cross_attn":
            feats = self.lin_attn(xz_features, yz_features)
            prediction = self.decoder(feats.view(bs * q, -1)).view(bs, q, -1)
        elif self.mode == "feat_concat":
            feats = torch.cat((xz_features, xz_features), dim = -1) 
            prediction = self.decoder(feats.view(bs * q, -1)).view(bs, q, -1)
        else: # vanilla
            xz_prediction = self.decoder(xz_features.view(bs * q, -1)).view(bs, q, -1)
            yz_prediction = self.decoder(yz_features.view(bs * q, -1)).view(bs, q, -1)
            # average between the two predictions
            prediction = (xz_prediction + yz_prediction) / 2
        return torch.sigmoid(prediction)

    def forward(self, xz_input, yz_input, coords, train = True):
        xz_latent_grid = self.encoder(xz_input)
        yz_latent_grid = self.encoder(yz_input)

        xz_features = self.grid.compute_features(xz_input, xz_latent_grid, coords)
        yz_features = self.grid.compute_features(yz_input, yz_latent_grid, coords)

        bs, q = coords.squeeze(1).squeeze(1).shape[:2]

        if self.mode == "cross_attn":
            features = self.lin_attn(xz_features, yz_features)
            prediction = self.decoder(features.view(bs * q, -1)).view(bs, q, -1)
        elif self.mode == "feat_concat":
            features = torch.cat((xz_features, yz_features), dim = -1)
            prediction = self.decoder(features.view(bs * q, -1)).view(bs, q, -1)
        else: # vanilla
            xz_prediction = self.decoder(xz_features.view(bs * q, -1)).view(bs, q, -1)
            yz_prediction = self.decoder(yz_features.view(bs * q, -1)).view(bs, q, -1)
            # average between the two predictions
            prediction = (xz_prediction + yz_prediction) / 2

        return torch.sigmoid(prediction)
