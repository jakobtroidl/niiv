import torch
from torch import nn
from niiv.feature_grid import FeatureGrid
from niiv.encoders.edsr_2d import EDSR2D
from niiv.decoder.mlp import MLP
from niiv.encoders import rdn
from niiv.encoders import swinir
from niiv.decoder import inr
from niiv.decoder.field_siren import FieldSiren



import torch
import torch.nn as nn

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # typically a scalar (e.g., 4, 7, etc.)
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        # Linear layer to project input to queries, keys, and values
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, H, W, C]
        Returns:
            out: Tensor of shape [B, H, W, C] after window attention.
        """
        B, H, W, C = x.shape
        ws = self.window_size
        # Ensure H and W are divisible by window size
        assert H % ws == 0 and W % ws == 0, "H and W must be divisible by window_size"

        # Partition input into non-overlapping windows
        x = x.view(B, H // ws, ws, W // ws, ws, C)  # [B, nH, ws, nW, ws, C]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # [B, nH, nW, ws, ws, C]
        windows = x.view(-1, ws * ws, C)  # [num_windows*B, ws*ws, C]

        # Compute queries, keys, and values
        qkv = self.qkv(windows)  # [num_windows*B, ws*ws, 3*C]
        # Split and reshape into [num_windows*B, num_heads, ws*ws, head_dim]
        qkv = qkv.reshape(windows.shape[0], windows.shape[1], 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, num_windows*B, num_heads, ws*ws, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention within each window
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [num_windows*B, num_heads, ws*ws, ws*ws]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Attention output
        out = (attn @ v)  # [num_windows*B, num_heads, ws*ws, head_dim]
        out = out.transpose(1, 2).reshape(windows.shape[0], ws * ws, C)

        # Final linear projection
        out = self.proj(out)
        out = self.proj_drop(out)

        # Merge windows back to original shape
        out = out.view(B, H // ws, W // ws, ws, ws, C)
        out = out.permute(0, 1, 3, 2, 4, 5).contiguous()
        out = out.view(B, H, W, C)
        return out

# # Example usage:
# if __name__ == '__main__':
#     B, H, W, C = 2, 8, 8, 64  # Batch size, Height, Width, Channels
#     window_size = 4
#     num_heads = 4
#     x = torch.randn(B, H, W, C)
#     win_attn = WindowAttention(dim=C, window_size=window_size, num_heads=num_heads, dropout=0.1)
#     out = win_attn(x)
#     print("Output shape:", out.shape)  # Expected: [B, H, W, C]


class Attention(nn.Module):
    def __init__(self, dim, n_heads=1):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)

    def forward(self, x, context):
        out, _ = self.attn(x, context, context)
        return out
    

def exists(val):
    return val is not None
    
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
    
class Decoder(nn.Module):
    def __init__(self, dim, n_heads=1, output_dim=128):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads

        self.attn = nn.ModuleList([
            PreNorm(dim, Attention(dim, n_heads=n_heads), context_dim = dim),
            PreNorm(dim, nn.Linear(dim, 1))
        ])

        self.queries = torch.nn.Parameter(torch.randn(1, output_dim**2, dim))
        # self.proj = nn.Linear(dim + 8, 1)
        self.proj = nn.Linear(dim, 1)


    def forward(self, x, coords, bilinear_queries):
        bs, q, _ = x.shape
        queries = self.queries.repeat(bs, 1, 1)
        cross_attn, cross_ff = self.attn
        queries = self.queries.expand(bs, -1, -1)
        out = cross_attn(queries, context = x) + queries
        out = cross_ff(out)
        return out


class NIIV(nn.Module):
    def __init__(self, out_features=1, encoding_config=None, n_pos_enc_octaves=2, **kwargs):
        super().__init__()

        self.feat_unfold = True

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
        self.attn = Attention(n_features)
        model_in = self.grid.n_out(n_features) 

        # self.wa = WindowAttention(dim=n_features, window_size=16, num_heads=1)


        # trainable parameters
        # self.encoder = rdn.make_rdn()
        # self.encoder = swinir.make_swinir()
        self.decoder = MLP(in_dim=model_in, out_dim=out_features, n_neurons=n_neurons, n_hidden=n_layers)
        # self.decoder = Decoder(n_features)

    def forward(self, image, coords):
        latent_grid = self.encoder(image)
        # latent_grid = latent_grid.permute(0, 2, 3, 1)
        # latent_grid = self.wa(latent_grid)
        # latent_grid = latent_grid.permute(0, 3, 1, 2)

        # features = self.grid.compute_features(image, latent_grid, coords, attn=self.attn)
        features = self.grid.compute_features(image, latent_grid, coords, self.attn)
        bs, q = coords.squeeze(1).squeeze(1).shape[:2]
        prediction = self.decoder(features.view(bs * q, -1)).view(bs, q, -1)
        return torch.sigmoid(prediction)
