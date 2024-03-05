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
