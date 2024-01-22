import os

import torch
from torch.utils.data import Dataset
import json
from PIL import Image
import torchvision.transforms as transforms

from util.utils import make_coord

class ImageDataset(Dataset):
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

        if self.is_train or self.isotropic_test_data:
            coords = make_coord(batched_gt.shape[-2:]).cuda()
            return [input, coords, gt_linear, file_name]
        else:
            output_shape = [batched_gt.shape[-2], int(batched_gt.shape[-1] * self.anisotropic_factor + 1)]
            coords = make_coord(output_shape).cuda()
        
        return [gt, coords, gt_linear, file_name]