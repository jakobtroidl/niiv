import torch
import numpy as np


class SIRENData():
    def __init__(self, path):
        """Load the volume in cfg.path into memory here."""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.avg_pool3D = torch.nn.AvgPool3d(kernel_size=[1, 1, int(8)])

        # load and normalize the volume
        self.gt = torch.from_numpy(np.load(path)).to(self.device)
        assert len(self.gt.shape) == 3
        self.gt = self.gt.unsqueeze(0).unsqueeze(0).to(torch.float32) # prepare for broadcasting
        self.gt = self.gt / 255 # normalize to [0, 1] 

        # 3D average pooling along z dimension
        self.input = self.avg_pool3D(self.gt).squeeze()


    def random_sample(self, batch_size):
        """Return a batch of random samples from the volume."""

        volume = self.input.squeeze()
        x = torch.randint(0, volume.shape[0], (batch_size,))
        y = torch.randint(0, volume.shape[1], (batch_size,))
        z = torch.randint(0, volume.shape[2], (batch_size,))

        coordinates = (
            torch.stack((x, y, z), dim=1).float() / torch.tensor(volume.shape).float()
        )
        coordinates = coordinates.to(self.device)
        values = volume[x, y, z].unsqueeze(1)

        return coordinates, values
    
    def sample_gt(self):
        X, Y, Z = self.gt_grid_size()

        # make coords for X, Y
        x = torch.arange(X).float() / X
        y = torch.arange(Y).float() / Y
        z = torch.arange(Z).float() / Z
        x, y, z = torch.meshgrid(x, y, z)
        coordinates = torch.stack((x, y, z), dim=0).to(self.device)

        # sample the gt
        return coordinates, self.gt



    def d_coordinate(self):
        return 3

    def d_out(self):
        return 1

    def input_grid_size(self):
        """Return a grid size that corresponds to the image's shape."""
        _, _, width, height, depth = self.input.shape
        return (width, height, depth)
    
    def gt_grid_size(self):
        """Return a grid size that corresponds to the image's shape."""
        _, _, width, height, depth = self.gt.shape
        return (width, height, depth)