from torch import nn
from niiv.decoder.sine_layer import SineLayer

class FieldSiren(nn.Module):
    network: nn.Sequential

    def __init__(
        self,
        d_coordinate: int,
        n_layers: int,
        n_neurons: int,
        d_out: int,
    ) -> None:
        """Set up a SIREN network using the sine layers"""
        super().__init__()

        layers = []
        layers.append(SineLayer(d_coordinate, n_neurons))
        for i in range(n_layers - 1):
            layers.append(SineLayer(n_neurons, n_neurons))
        layers.append(nn.Linear(n_neurons, d_out))

        self.model = nn.Sequential(*layers)

    def forward(self, coordinates):
        """Evaluate the MLP at the specified coordinates."""
        return self.model(coordinates)
