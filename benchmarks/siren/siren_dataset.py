import torch
import torch.nn.functional as F


class FieldDatasetVolume():
    def __init__(self, cfg):
        """Load the volume in cfg.path into memory here."""

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # torch load tensor from file
        data = torch.load(cfg.path)
        assert len(data.shape) == 3

        # normalize to [0, 1]
        # data = data / 255
        # data = data * 2 - 1

        data = data.unsqueeze(0).unsqueeze(0)  # prepare for broadcasting

        min = data.min()
        max = data.max()
        print(f"min: {min}, max: {max}")
        print(f"shape: {data.shape}")
        print(f"type: {data.dtype}")

        self.volume = data.to(self.device)

    def random_sample(self, batch_size):
        """Return a batch of random samples from the volume."""

        volume = self.volume.squeeze()

        x = torch.randint(0, volume.shape[0], (batch_size,))
        y = torch.randint(0, volume.shape[1], (batch_size,))
        z = torch.randint(0, volume.shape[2], (batch_size,))

        coordinates = (
            torch.stack((x, y, z), dim=1).float() / torch.tensor(volume.shape).float()
        )
        coordinates = coordinates.to(self.device)
        values = volume[x, y, z].unsqueeze(1)

        return coordinates, values

    def query(self, coordinates):
        """Sample the image at the specified coordinates and return the corresponding
        colors. Remember that the coordinates will be in the range [0, 1].

        You may find the grid_sample function from torch.nn.functional helpful here.
        Pay special attention to grid_sample's expected input range for the grid
        parameter.
        """

        batch_size = coordinates.shape[0]

        coordinates = coordinates * 2 - 1  # normalize coordinates to [-1, 1]
        coords = coordinates.view(1, batch_size, 1, 1, 3).to(self.device)

        output = F.grid_sample(
            self.volume,
            coords,
            align_corners=True,
        )

        output = output.squeeze().unsqueeze(1)

        return output

    def d_coordinate(self) -> int:
        return 3

    def d_out(self) -> int:
        return 1

    def grid_size(self):
        """Return a grid size that corresponds to the image's shape."""
        _, _, width, height, depth = self.volume.shape
        return (width, height, depth)