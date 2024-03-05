import torch
from niiv.util.utils import make_coord
from niiv.decoder.pos_enc import PositionalEncoding

class FeatureGrid():
    def __init__(self, feat_unfold, n_pos_encoding):
        self.feature_unfold = feat_unfold
        self.pos_enc = PositionalEncoding(num_octaves=n_pos_encoding)
    
    def n_out(self, n_in):
        if self.feature_unfold:
            return n_in * 9
        return n_in + self.pos_enc.d_out(2) + 1
        # return n_in + 2 + 1
    
    def compute_features(self, image, latents, coords):

        if self.feature_unfold:
            # concat each latent by it's local neighborhood
            latents = self.unfold_features(latents)

        # interpolate feature coordinates
        feature_coords = make_coord(latents.shape[-2:], flatten=False).cuda()
        feature_coords = feature_coords.permute(2, 0, 1).unsqueeze(0)
        feature_coords = feature_coords.repeat(coords.shape[0], 1, 1, 1)

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

        pe_coords = self.pos_enc((q_coords + 1.0) / 2.0)

        # pe_coords = q_coords

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