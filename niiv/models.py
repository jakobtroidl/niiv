import torch
from torch import nn
from niiv.feature_grid import FeatureGrid
from niiv.encoders.edsr_2d import EDSR2D
from niiv.decoder.mlp import MLP
from niiv.encoders import rdn
from niiv.encoders import swinir
from niiv.decoder import inr
from niiv.decoder.field_siren import FieldSiren
import torch.nn.functional as F
from timm.models.layers import DropPath
import numpy as np


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, drop_path_rate = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.net(x))
    
class CrossAttention(nn.Module):
    def __init__(self, dim, n_heads = 1, dropout = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        
        # # feature normalization layers
        # self.norm1 = nn.LayerNorm(dim)
        # self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, q, k, v):
        # x = self.attn(self.norm2(q), self.norm1(k), self.norm1(v))[0]
        x, weights = self.attn(q, k, v)
        # print("weights min, max: ", weights.min(), weights.max())
        return x

class SelfAttention(nn.Module):
    def __init__(self, n_features, n_heads = 1, dropout = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=n_features, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ff = FeedForward(n_features, drop_path_rate = 0.1)
        
        # feature normalization layers
        self.norm1 = nn.LayerNorm(n_features)
        self.norm2 = nn.LayerNorm(n_features)
    
    def forward(self, qkv):
        x = qkv + self.attn(self.norm1(qkv), self.norm1(qkv), self.norm1(qkv))[0]
        x = x + self.ff(self.norm2(x))
        return x

class PointEmbed(nn.Module):

    def __init__(self, hidden_dim=88, dim=128):  # Adjusted to be divisible by 4
        super().__init__()

        assert hidden_dim % 4 == 0, "hidden_dim must be divisible by 4 for 2D input."

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 4)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 4)]),
            torch.cat([torch.zeros(self.embedding_dim // 4), e]),
        ])  # Shape: 2 x (hidden_dim // 4)

        self.register_buffer('basis', e)  # 2 x (hidden_dim // 4)

        self.mlp = nn.Linear(self.embedding_dim + 2, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum('bnd,de->bne', input, basis)  # B x N x embedding_dim/2
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)  # B x N x embedding_dim
        return embeddings

    def forward(self, input):
        # input: B x N x 2
        pe = self.embed(input, self.basis)
        embed = self.mlp(torch.cat([pe, input], dim=2))  # B x N x C
        return embed



class NIIV(nn.Module):
    def __init__(self, out_features=1, encoding_config=None, n_pos_enc_octaves=10, **kwargs):
        super().__init__()

        self.feat_unfold = False
        self.depth = 3
        self.attn_layers = []

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

        # self.attn = nn.MultiheadAttention(embed_dim=n_features, num_heads=1, dropout=0.1, batch_first=True)  
        # self.point_embed = PointEmbed(hidden_dim=80, dim=n_features)

        # for i in range(self.depth):
        #     self.attn_layers.append(SelfAttention(n_features))

        # self.attn_layers = nn.Sequential(*self.attn_layers)

        # module for latent grid processing
        self.grid = FeatureGrid(feat_unfold=self.feat_unfold, n_pos_encoding=n_pos_enc_octaves, n_features=n_features)

        self.cross_attn = CrossAttention(dim=n_features, n_heads=1, dropout=0.0)

        self.query_tokens = nn.Parameter(torch.randn(1, 128 * 128, n_features))


        # model_in = self.grid.n_out(n_features) 

        # trainable parameters
        # self.encoder = rdn.make_rdn()
        # self.encoder = swinir.make_swinir()
        self.decoder = MLP(in_dim=self.grid.n_out(n_features), out_dim=out_features, n_neurons=n_neurons, n_hidden=n_layers)

    def forward(self, image, coords):
        latent_grid = self.encoder(image)

        # coords = self.grid.comp_coords(image, latent_grid, coords)
        # emb_coords = self.coord_embed(coords)

        B, C, H, W = latent_grid.shape
        latent_grid = latent_grid.permute(0, 2, 3, 1)
        latent_grid = latent_grid.reshape(B, H * W, C)
        # latent_grid = self.attn_layers(latent_grid)

        # compute features using bilinear interpolation
        latent_grid_ = latent_grid.reshape(B, H, W, C).permute(0, 3, 1, 2)        
        _, q_input, pe_coords = self.grid.compute_features(image, latent_grid_, coords)

        queries = self.query_tokens.expand(B, -1, -1)  # (B, 128*128, C)

        # cross attention
        features = self.cross_attn(queries, latent_grid, latent_grid)
        features = torch.cat([features, q_input, pe_coords], dim=-1)

        prediction = self.decoder(features.reshape(B * W * W, -1)).reshape(B, W * W, -1)
        return torch.sigmoid(prediction)
