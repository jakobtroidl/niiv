import torch
from util.utils import make_coord

class FeatureGrid():
    def __init__(self, feat_unfold, local_ensemble, pos_encoding, upsample=False):
        self.latents = None
        self.upsample = upsample
        self.local_ensemble = local_ensemble
        self.feature_unfold = feat_unfold
        self.pos_enc = pos_encoding
    
    def compute_features(self, image, latents, coords, decoder):

        if self.feature_unfold:
            # concat each latent by it's local neighborhood
            latents = self.unfold_features(latents)

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / latents.shape[-2] / 2
        ry = 2 / latents.shape[-1] / 2

        # interpolate feature coordinates
        feature_coords = make_coord(latents.shape[-2:], flatten=False).cuda()
        feature_coords = feature_coords.permute(2, 0, 1).unsqueeze(0)
        feature_coords = feature_coords.repeat(coords.shape[0], 1, 1, 1)

        predictions = []
        areas = []

        coords = coords.unsqueeze(1)
        for vx in vx_lst:
            for vy in vy_lst:
                coords_ = coords.clone()
                coords_[..., 0] += vx * rx + eps_shift
                coords_[..., 1] += vy * ry + eps_shift
                coords_.clamp_(-1 + 1e-6, 1 - 1e-6)

                # interpolate features
                q_features = torch.nn.functional.grid_sample(latents, coords_.flip(-1), mode='bilinear', align_corners=False)
                q_features = q_features.squeeze(2).squeeze(2)
                q_features = q_features.permute(0, 2, 1)

                q_coords = torch.nn.functional.grid_sample(feature_coords, coords_.flip(-1), mode='bilinear', align_corners=False)
                q_coords = q_coords.squeeze(2).squeeze(2)
                q_coords = q_coords.permute(0, 2, 1)

                q_input = torch.nn.functional.grid_sample(image, coords_.flip(-1), mode='bilinear', align_corners=False)
                q_input = q_input.squeeze(2).squeeze(2)
                q_input = q_input.permute(0, 2, 1)

                rel_coord = coords_.squeeze(1).squeeze(1) - q_coords

                rel_coord[..., 0] *= latents.shape[-2]
                rel_coord[..., 1] *= latents.shape[-1]

                 # compute area for ensemble
                area = torch.abs(rel_coord[..., 0] * rel_coord[..., 1])
                areas.append(area + 1e-9)

                rel_coord = self.pos_enc(rel_coord)
                pe_coords = self.pos_enc((q_coords + 1.0) / 2.0)

                input = torch.cat((q_features, q_input, rel_coord, pe_coords), dim=-1)
                bs, q = coords.squeeze(1).squeeze(1).shape[:2]

                # compute prediction for ensemble
                prediction = decoder(input.view(bs * q, -1)).view(bs, q, -1)
                predictions.append(prediction)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        out = 0
        for pred, area in zip(predictions, areas):
            out = out + pred * (area / tot_area).unsqueeze(-1)

        out = torch.clamp(out, 0, 1)
        return out
    
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