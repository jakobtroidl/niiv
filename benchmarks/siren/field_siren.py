from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor, nn

from benchmarks.siren.sine_layer import SineLayer
from benchmarks.siren.field import Field


class FieldSiren(Field):
    network: nn.Sequential

    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up a SIREN network using the sine layers at src/components/sine_layer.py.
        Your network should consist of:

        - An input sine layer whose output dimensionality is 256
        - Two hidden sine layers with width 256
        - An output linear layer
        """
        super().__init__(cfg, d_coordinate, d_out)

        layers = []

        layers.append(SineLayer(d_coordinate, 256))
        layers.append(SineLayer(256, 256))
        layers.append(SineLayer(256, 256))
        layers.append(nn.Linear(256, d_out))

        self.model = nn.Sequential(*layers)

    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        """Evaluate the MLP at the specified coordinates."""

        return self.model(coordinates)