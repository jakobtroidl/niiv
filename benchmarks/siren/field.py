from torch import nn
import torch
from benchmarks.siren.sine_layer import SineLayer
class FieldSiren(torch.nn.Module):
    def __init__(self, opt):
        """Set up a SIREN network using the sine layers at src/components/sine_layer.py.
        Your network should consist of:

        - An input sine layer whose output dimensionality is 256
        - Two hidden sine layers with width 256
        - An output linear layer
        """
        super().__init__()

        net = opt["network"]
        layers = []
        layers.append(SineLayer(net["in_dim"], net["n_neurons"]))
        for _ in range(net["n_hidden_layers"]):
            layers.append(SineLayer(net["n_neurons"], net["n_neurons"]))
        layers.append(nn.Linear(net["n_neurons"], net["out_dim"]))
        self.model = nn.Sequential(*layers)

    def forward(self, coordinates):
        """Evaluate the MLP at the specified coordinates."""
        return self.model(coordinates)