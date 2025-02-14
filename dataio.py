import os

import torch
from torch.utils.data import Dataset
import json
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random
import seaborn as sns
import torchvision.transforms.functional as TF


from niiv.util.utils import make_coord
import pandas as pd
import matplotlib.pyplot as plt

def save_ablation_linechart(path, x, y_result, y_bilinear=None, y_nearest=None):
    df = pd.DataFrame({
        'Threshold': x,
        'Ours': y_result,
        'Bilinear': y_bilinear,
        'Nearest': y_nearest
    })

    df_long = pd.melt(df, id_vars=['Threshold'], var_name='Category', value_name='CF PSNR')
    sns.set_style("darkgrid")
    sns.lineplot(data=df_long, x='Threshold', y='CF PSNR', hue='Category')

    output_path = os.path.join(path, "cfpsnr_lineplot.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    plt.clf()

def create_dir(path, folder):
    path = os.path.join(path, folder)
    if not os.path.exists(path):
        os.makedirs(path)
    return path 

def save_images(path, images, names=None, metrics=None, metric_idx=None):
    for i in range(images.shape[0]):
        image = images[i, ...].squeeze()
        if names is None:
            name = str(i) + ".png"
        else:
            name = str(names[i])
        if metrics is not None:
            splits = name.split(".")
            metric = str(metrics[i][metric_idx].item())
            metric = metric.replace(".", "_")
            name = "{}_{}.{}".format(splits[0], metric, splits[1])
        image = image.detach().cpu().numpy()
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
        image.save(os.path.join(path, name))


class ImageDatasetTest(Dataset):
    def __init__(self, path_to_info, name, x_or_y=0) -> None:
        super().__init__()
        info = json.loads(open(path_to_info).read())
        self.path = os.path.join(os.path.dirname(path_to_info), "test")
        self.path = os.path.join(self.path, name)
        assert x_or_y == 0 or x_or_y == 1
        self.x_or_y = x_or_y
        self.isotropic_test_data = bool(info["isotropic_test_data"])
        self.anisotropic_factor = info["anisotropic_factor"]

        self.avg_pool = torch.nn.AvgPool3d(kernel_size=[1, 1, int(self.anisotropic_factor)])
        self.data = torch.from_numpy(np.load(self.path)).cuda()
        self.data = self.data.to(torch.float32) / 255.0
        if self.isotropic_test_data:
            self.anisotropic = self.avg_pool(self.data.unsqueeze(0).unsqueeze(0)).squeeze()
        else:
            self.anisotropic = self.data[:, :, :self.data.shape[-1]//self.anisotropic_factor]

    def has_isotropic_test_data(self):
        return self.isotropic_test_data

    def __len__(self):
        idx = int(not bool(self.x_or_y))
        return self.anisotropic.shape[-3 + idx]
    
    def __getitem__(self, idx):
        
        xz_input = self.anisotropic[:, idx, :]
        xz_input = xz_input.permute(1, 0) # align to 1, 16, 128 shape 
        xz_gt = self.data[:, idx, :]
        xz_gt = xz_gt.permute(1, 0)
        xz_name = "xz_"

        yz_input = self.anisotropic[idx, :, :]
        yz_gt = self.data[idx, :, :]
        yz_name = "yz_"

        xz_name += str(xz_name) + str(idx) + ".png" 
        yz_name += str(yz_name) + str(idx) + ".png"

        xz_input = xz_input.unsqueeze(0)
        xz_gt_linear = xz_gt.unsqueeze(-1)
        xz_gt_linear = xz_gt_linear.reshape(-1, 1)
        coords = make_coord(xz_gt.shape[-2:]).cuda()

        # # safe xz_input, xz_gt as png file
        # image = TF.to_pil_image(xz_input)
        # image.save("test_xz_input.png")

        # image = TF.to_pil_image(xz_gt)
        # image.save("test_xz_gt.png")

        return [xz_input, coords, xz_gt_linear, xz_name]

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
        self.anisotropic_factor = info["anisotropic_factor"]

        # self.avg_pool2D = torch.nn.AvgPool2d(kernel_size=[1, int(self.anisotropic_factor)])
        self.avg_pool3D = torch.nn.AvgPool3d(kernel_size=[1, 1, int(self.anisotropic_factor)])

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

        if self.isotropic_test_data:
            anisotropic = self.avg_pool3D(gt.unsqueeze(0)).squeeze()
        else:
            anisotropic = gt[:, :, :gt.shape[-1]//self.anisotropic_factor]

        # slice_idx = np.random.randint(0, anisotropic.shape[-3 + self.x_or_y])
        xy_idx = np.random.randint(0, anisotropic.shape[-1])
        xy = anisotropic[:, :, xy_idx]

        x_degrador = torch.nn.AvgPool2d(kernel_size=[int(self.anisotropic_factor), 1])
        y_degrador = torch.nn.AvgPool2d(kernel_size=[1, int(self.anisotropic_factor)])
        
        xy_gt = xy.unsqueeze(0)
        x_degraded_input = x_degrador(xy_gt)
        y_degraded_input = y_degrador(xy_gt)
        
        y_degraded_input = y_degraded_input.permute(0, 2, 1)
        xy_coords = make_coord(xy.shape[-2:]).cuda()

        # # safe x_degraded_input, y_degraded_input as png file
        # image = TF.to_pil_image(x_degraded_input)
        # image.save("x_degraded.png")

        # image = TF.to_pil_image(y_degraded_input)
        # image.save("y_degraded.png")

        # image = TF.to_pil_image(xy_gt)
        # image.save("xy_gt.png")
        

        output = {
            "x_degraded": [x_degraded_input, xy_coords, xy_gt], # xy slice degraded along x
            "y_degraded": [y_degraded_input, xy_coords, xy_gt], # xy slice degraded along y
        }
        
        return output