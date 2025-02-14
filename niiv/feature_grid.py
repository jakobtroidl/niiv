import torch
import torch.nn as nn

from niiv.util.utils import make_coord
from niiv.decoder.pos_enc import PositionalEncoding
from timm.models.layers import DropPath
from torch import nn, einsum
from einops import rearrange, repeat

def default(val, d):
    return val if exists(val) else d

def exists(val):
    return val is not None

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

class FeatureGrid(nn.Module):
    def __init__(self, feat_unfold, n_pos_encoding, n_features):
        super().__init__()

        # initialize variables
        self.n_features = n_features
        self.feature_unfold = feat_unfold
        self.pos_enc = PositionalEncoding(num_octaves=n_pos_encoding)
        self.feat_attn = PreNorm(n_features, Attention(n_features, n_features, heads=1, dim_head=n_features, drop_path_rate=0.1), context_dim=n_features)
        # self.input_attn = nn.MultiheadAttention(embed_dim=1, num_heads=1, batch_first=True)
    
    def n_out(self, n_in):
        if self.feature_unfold:
            return n_in * 9
        return n_in + self.pos_enc.d_out(2) + 1
        # return n_in + 2 + 1

    def nearest_features(self, grid_features, grid_coords, query_coords, N):
        """
        Finds the N nearest features for each query coordinate.

        Args:
            grid_features (torch.Tensor): Feature grid of shape (H, W, C) where C is the feature dimension.
            grid_coords (torch.Tensor): 2D coordinates of shape (H, W, 2) corresponding to grid positions.
            query_coords (torch.Tensor): Query coordinates of shape (Q, 2) where Q is the number of query points.
            N (int): Number of nearest neighbors to find.

        Returns:
            torch.Tensor: Nearest features of shape (Q, N, C).
            torch.Tensor: Distances of shape (Q, N).
        """
        grid_features = grid_features.permute(0, 2, 3, 1)
        grid_coords = grid_coords.permute(0, 2, 3, 1)
        B, H, W, C = grid_features.shape
        
        grid_coords_flat = grid_coords.reshape(B, -1, 2)
        distances = torch.cdist(query_coords, grid_coords_flat, p=2)
        nearest_indices = torch.topk(distances, k=N, largest=False).indices  # Shape: (Q, N)       
        
        grid_features_flat = grid_features.reshape(-1, C)  # Shape: (H*W, C)
        nearest_features = grid_features_flat[nearest_indices]  # Shape: (Q, N, C)

        nearest_dists = distances.gather(1, nearest_indices)

        # sort the nearest features by distance to the query point
        idx = torch.argsort(nearest_dists, dim=-1)
        nearest_features = nearest_features[idx]

        return nearest_features

    
    def compute_features(self, image, latents, coords, use_attn=True, K = 9):

        if self.feature_unfold:
            # concat each latent by it's local neighborhood
            latents = self.unfold_features(latents)

        # interpolate feature coordinates
        feature_coords = make_coord(latents.shape[-2:], flatten=False).cuda()
        feature_coords = feature_coords.permute(2, 0, 1).unsqueeze(0)
        feature_coords = feature_coords.repeat(coords.shape[0], 1, 1, 1)

        # compute 
        # nearest_features = self.nearest_features(latents, feature_coords, coords, K)
        # B, N, K, C = nearest_features.shape
        # nearest_features = nearest_features.view(B * N, K, C)

        coords_ = coords.unsqueeze(1)
        # interpolate features
        q_features = torch.nn.functional.grid_sample(latents, coords_.flip(-1), mode='bilinear', align_corners=None)
        q_features = q_features.squeeze(2).squeeze(2)
        q_features = q_features.permute(0, 2, 1)

        q_coords = torch.nn.functional.grid_sample(feature_coords, coords_.flip(-1), mode='bilinear', align_corners=None)
        q_coords = q_coords.squeeze(2).squeeze(2)
        q_coords = q_coords.permute(0, 2, 1)

        q_input = torch.nn.functional.grid_sample(image, coords_.flip(-1), mode='bilinear', align_corners=None)
        q_input = q_input.squeeze(2).squeeze(2)
        q_input = q_input.permute(0, 2, 1)

        # if use_attn:
        #     q_features = q_features.reshape(B * N, C).unsqueeze(1)
        #     attn_out = self.feat_attn(q_features, context = nearest_features)
        #     q_features = attn_out.squeeze(1).reshape(B, N, C)

        pe_coords = self.pos_enc((q_coords + 1.0) / 2.0)

        decoder_input = torch.cat((q_features, q_input, pe_coords.squeeze()), dim=-1)
        return decoder_input
    
    def unfold_features(self, latents):
        unfold_list = [-1, 0, 1]
        bs, n_feat, x_dim, y_dim = latents.shape

        x_idx = torch.arange(0, latents.shape[-2], dtype=torch.int64).cuda()
        y_idx = torch.arange(0, latents.shape[-1], dtype=torch.int64).cuda()

        tmp_x, tmp_y = torch.meshgrid(x_idx, y_idx)

        idx = torch.stack((tmp_x, tmp_y), dim=-1)

        x = idx[..., 0].flatten()
        y = idx[..., 1].flatten()

        neighbors = []

        for i in unfold_list:
            for j in unfold_list:
                vx = torch.clamp(x+i, 0, latents.shape[-2]-1)
                vy = torch.clamp(y+j, 0, latents.shape[-1]-1)

                nbh = latents[:, :, vx, vy]
                neighbors.append(nbh)

        latents = torch.stack(neighbors, dim=1).view(bs, n_feat * len(unfold_list)**2, x_dim, y_dim)
        return latents