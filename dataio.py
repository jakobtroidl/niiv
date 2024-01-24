import os

import torch
from torch.utils.data import Dataset
import json
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random
from skimage.restoration import denoise_tv_chambolle


from util.utils import make_coord


class ImageDatasetTest(Dataset):
    def __init__(self, path_to_info, train=True, folder=None) -> None:
        super().__init__()
        info = json.loads(open(path_to_info).read())
        self.is_train = train
        self.folder = folder
        self.path = os.path.join(os.path.dirname(path_to_info), "train" if self.is_train else "test_sequence")

        if self.folder is not None:
            self.path = os.path.join(self.path, self.folder)

        self.files = os.listdir(self.path)
        self.anisotropic_factor = 7.5  # TODO make dynamic
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=[1, int(self.anisotropic_factor)])
        self.isotropic_test_data = info["isotropic_test_data"]

    def has_isotropic_test_data(self):
        return self.isotropic_test_data

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_name = self.files[idx]
        path = os.path.join(self.path, file_name)

        image = Image.open(path) # Load the image
        transform = transforms.ToTensor() # Transform to tensor
        gt = transform(image).cuda() # Transform to tensor, already in [0, 1]

        batched_gt = gt.unsqueeze(0)
        input = self.avg_pool(batched_gt)
        input = input.squeeze(0)

        coords = make_coord(batched_gt.shape[-2:]).cuda()

        gt_linear = gt.permute(1, 2, 0)
        gt_linear = gt_linear.view(-1, gt_linear.shape[-1])


        coords = make_coord(batched_gt.shape[-2:]).cuda()
        return [input, coords, gt_linear, file_name]
        


class ImageDataset(Dataset):
    def __init__(self, path_to_info, train=True, folder=None) -> None:
        super().__init__()
        info = json.loads(open(path_to_info).read())
        self.is_train = train
        self.folder = folder
        self.path = os.path.join(os.path.dirname(path_to_info), "train" if self.is_train else "test_sequence")
        self.denoise = True

        if self.folder is not None:
            self.path = os.path.join(self.path, self.folder)

        self.files = os.listdir(self.path)
        self.anisotropic_factor = 7.5  # TODO make dynamic
        # self.avg_pools = [  torch.nn.AvgPool2d(kernel_size=[1, int(self.anisotropic_factor)]), 
        #                     torch.nn.AvgPool2d(kernel_size=[int(self.anisotropic_factor), 1])]

        self.avg_pools = torch.nn.AvgPool2d(kernel_size=[1, int(self.anisotropic_factor)])

        self.isotropic_test_data = info["isotropic_test_data"]
        self.x_or_y = random.randint(0, 1)
    
    def shuffle_x_or_y(self):
        self.x_or_y = random.randint(0, 1)

    def has_isotropic_test_data(self):
        return self.isotropic_test_data

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_name = self.files[idx]
        path = os.path.join(self.path, file_name)

        image = np.load(path)
        transform = transforms.ToTensor() # Transform to tensor
        gt = transform(image).cuda() # Transform to tensor, already in [0, 1]

        slice_idx = np.random.randint(0, gt.shape[-2 + self.x_or_y])
        xy_idx = np.random.randint(0, gt.shape[0])

        xy = gt[:, :, xy_idx]

        if self.x_or_y == 1:
            slice = gt[slice_idx, :, :]
        else:
            slice = gt[:, slice_idx, :]

        xy_gt = xy.unsqueeze(0)
        slice_gt = slice.unsqueeze(0)
        if self.denoise: 
            xy_gt_np = xy_gt.cpu().detach().numpy()
            xy_gt_np = denoise_tv_chambolle(xy_gt_np, weight=0.1, channel_axis=0)
            xy_gt = torch.from_numpy(xy_gt_np).cuda()

            slice_gt_np = slice_gt.cpu().detach().numpy()
            slice_gt_np = denoise_tv_chambolle(slice_gt_np, weight=0.1, channel_axis=0)
            slice_gt = torch.from_numpy(slice_gt_np).cuda()

        xy_inputs = self.avg_pools(xy_gt)
        xy_coords = make_coord(xy.shape[-2:]).cuda()
        
        slice_inputs = self.avg_pools(slice_gt)
        slice_coords = make_coord(slice.shape[-2:]).cuda()

        output = {
            "xy": [xy_inputs, xy_coords, xy_gt],
            "slice": [slice_inputs, slice_coords, slice_gt],
            "meta": [self.x_or_y, xy_idx, slice_idx]
        }
        
        return output