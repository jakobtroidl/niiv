import torch
from torch import nn
from src.feature_grid import FeatureGrid
from src.encoders.edsr_2d import EDSR2D
from src.decoder.mlp import MLP
from src.encoders import rdn
from src.encoders import swinir
from src.decoder import inr
from src.decoder.field_siren import FieldSiren

class NIV(nn.Module):
    def __init__(self, out_features=3, encoding_config=None, latent_grid=None, n_pos_enc_octaves=10, **kwargs):
        super().__init__()

        self.feat_unfold = False

        # hyper parameters
        n_features = encoding_config["encoder"]["n_features"]
        n_layers = encoding_config["network"]["n_hidden_layers"]
        n_neurons = encoding_config["network"]["n_neurons"]   

        # module for latent grid processing
        self.latent_grid = latent_grid
        self.grid = FeatureGrid(feat_unfold=self.feat_unfold, n_pos_encoding=n_pos_enc_octaves, upsample=False)
        model_in = self.grid.n_out(n_features) 

        # trainable parameters
        self.encoder = EDSR2D(args = encoding_config["encoder"])
        # self.encoder = rdn.make_rdn()
        # self.encoder = swinir.make_swinir()
        self.decoder = MLP(in_dim=model_in, out_dim=out_features, n_neurons=n_neurons, n_hidden=n_layers)

    def forward(self, image, coords):
        latent_grid = self.encoder(image)
        features = self.grid.compute_features(image, latent_grid, coords)
        bs, q = coords.squeeze(1).squeeze(1).shape[:2]
        prediction = self.decoder(features.view(bs * q, -1)).view(bs, q, -1)
        return torch.clamp(prediction, 0, 1)
