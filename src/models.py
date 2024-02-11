from torch import nn
from src.feature_grid import FeatureGrid
from src.encoders.edsr_2d import EDSR2D
from src.decoder.mlp import MLP
from src.encoders import rdn
from src.encoders import swinir
from src.decoder import inr
from src.decoder.field_siren import FieldSiren
from src.decoder.pos_enc import PositionalEncoding

class NIV(nn.Module):
    def __init__(self, out_features=3, encoding_config=None, latent_grid=None, export_features=False, pos_enc=True, **kwargs):
        super().__init__()

        self.feat_unfold = False
        self.local_ensemble = True
        
        if pos_enc:
            self.pos_enc = PositionalEncoding(num_octaves=10)
        else:
            self.pos_enc = None

        self.export_features = export_features

        # hyper parameters
        n_features = encoding_config["encoder"]["n_features"]
        n_layers = encoding_config["network"]["n_hidden_layers"]
        n_neurons = encoding_config["network"]["n_neurons"]


        if self.pos_enc:
            n_pos =  self.pos_enc.d_out(2) * 2
        else:
            n_pos = 2 

        if self.feat_unfold:
            feat_dim = 3**2 # expand features by local neighborhood
        else:
            feat_dim = 1

        model_in = n_features * feat_dim + n_pos + 1

        # module for latent grid processing
        self.latent_grid = latent_grid
        self.sparse_grid = FeatureGrid(feat_unfold=self.feat_unfold, local_ensemble=self.local_ensemble, pos_encoding=self.pos_enc, upsample=False)

        # trainable parameters
        self.encoder = EDSR2D(args = encoding_config["encoder"])
        # self.encoder = rdn.make_rdn()
        # self.encoder = swinir.make_swinir()
        self.decoder = MLP(in_dim=model_in, out_dim=out_features, n_neurons=n_neurons, n_hidden=n_layers, pos_enc=self.pos_enc)
        # self.decoder = inr.Gabor(in_features=model_in, hidden_features=mlp_n_neurons, hidden_layers=mlp_n_layers, out_features=out_features)
        # self.decoder = FieldSiren(d_coordinate=model_in, d_out=out_features, n_layers=n_layers, n_neurons=n_neurons)

    def forward(self, image, coords):
        latent_grid = self.encoder(image)
        return self.sparse_grid.compute_features(image, latent_grid, coords, self.decoder)
