import torch
from torch import nn
from niiv.feature_grid import FeatureGrid
from niiv.encoders.edsr_2d import EDSR2D
from niiv.decoder.mlp import MLP
from niiv.encoders import rdn
from niiv.encoders import swinir
from niiv.decoder import inr
from niiv.decoder.field_siren import FieldSiren

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
        self.grid = FeatureGrid(feat_unfold=self.feat_unfold, n_pos_encoding=n_pos_enc_octaves, n_features=n_features)
        model_in = self.grid.n_out(n_features) 

        # trainable parameters
        # self.encoder = rdn.make_rdn()
        # self.encoder = swinir.make_swinir()
        self.decoder = MLP(in_dim=model_in, out_dim=out_features, n_neurons=n_neurons, n_hidden=n_layers)

    def fwd_train(self, im1, im2, coords):

        B, N, _ = coords.shape
        im_size = int(N ** 0.5)

        f1 = self.encode(im1, coords)
        f2 = self.encode(im2, coords)

        # rotate feature spaces for alignment
        f2 = f2.view(-1, im_size, im_size, f2.shape[-1])
        f2 = f2.permute(0, 2, 1, 3)
        f2 = f2.reshape(B, N, f2.shape[-1])

        f = torch.stack([f1, f2], dim=-1)
        f = torch.mean(f, dim=-1)

        pred = self.decode(f, coords)

        pred_f1 = self.decode(f1, coords)
        pred_f2 = self.decode(f2, coords)

        return pred, f1, f2, pred_f1, pred_f2


    
    def encode(self, image, coords):
        latent_grid = self.encoder(image)
        features = self.grid.compute_features(image, latent_grid, coords)
        return features
    
    def decode(self, features, coords):
        bs, q = coords.squeeze(1).squeeze(1).shape[:2]
        prediction = self.decoder(features.view(bs * q, -1)).view(bs, q, -1)
        return torch.sigmoid(prediction)

    def forward(self, image, coords):
        latent_grid = self.encoder(image)
        features = self.grid.compute_features(image, latent_grid, coords)
        bs, q = coords.squeeze(1).squeeze(1).shape[:2]
        prediction = self.decoder(features.view(bs * q, -1)).view(bs, q, -1)
        return torch.sigmoid(prediction), features
