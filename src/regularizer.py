import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.image import image_gradients
from util.transforms import crop_image_border
import matplotlib.pyplot as plt
from util.eval_metrics import FourierDenoiser


class GradientRegularizer(nn.Module):
    def __init__(self):
        super(GradientRegularizer, self).__init__()
        self.sobel_x = torch.tensor([[-0.25, 0, 0.25], [-0.5, 0, 0.5], [-0.25, 0, 0.25]], dtype=torch.float32).view(1, 1, 3, 3).cuda()
        self.sobel_y = torch.tensor([[-0.25, -0.5, -0.25], [0, 0, 0], [0.25, 0.5, 0.25]], dtype=torch.float32).view(1, 1, 3, 3).cuda()
        self.denoiser = FourierDenoiser(threshold=25)


    def forward(self, x, epoch, step, weight=0.01):
        x = self.denoiser(x)
        x = x.unsqueeze(1)
        # dy, dx = image_gradients(x)

        x_padded = F.pad(x, (1, 1, 1, 1), mode='replicate')

        G_x = F.conv2d(x_padded, self.sobel_x)
        G_y = F.conv2d(x_padded, self.sobel_y)

        mag = torch.sqrt(G_x**2 + G_y**2 + 1e-9)
        # mag = mag - 0.05
        # mag = torch.clamp(mag, 0.0, 0.35) # disallow noise like and large gradients to dominate the loss
        # mag = crop_image_border(mag, 5)


        # if epoch % 5 == 0 and step == 0 :
        #     ## plot histogram of dy, dx and mag
        #     plt.hist(dy.flatten().detach().cpu().numpy(), bins=100)
        #     plt.savefig('histograms/dy-histogram-{}.png'.format(epoch)) 
        #     plt.clf()   
        #     plt.hist(dx.flatten().detach().cpu().numpy(), bins=100)
        #     plt.savefig('histograms/dx-histogram-{}.png'.format(epoch))
        #     plt.clf()
        #     plt.hist(mag.flatten().detach().cpu().numpy(), bins=100)
        #     plt.savefig('histograms/mag-histogram-{}.png'.format(epoch))
        #     plt.clf()

        x = torch.pow(1.0 - torch.mean(torch.abs(mag)), 1)
        x = torch.clamp(x, 0.0, 1.0)
        return x * weight


class FourierRegularizer(nn.Module):
    def __init__(self, filter_radius=25):
        super(FourierRegularizer, self).__init__()
        self.filter_radius = filter_radius

    def forward(self, x):
        # Compute the 2D Fourier Transform of the image
        f_transform = torch.fft.fft2(x)
        f_shifted = torch.fft.fftshift(f_transform)

        # Create a low-pass filter mask (circular mask)
        rows, cols = x.shape[-2:]
        crow, ccol = rows // 2 , cols // 2
        low_pass = torch.zeros((rows, cols), dtype=torch.uint8)
        y, x = torch.meshgrid[:rows, :cols]
        mask_area = (x - ccol) ** 2 + (y - crow) ** 2 <= self.filter_radius**2
        low_pass[mask_area] = 1

        # Apply the mask/filter
        f_shifted_filtered = f_shifted * low_pass

        # Inverse Fourier Transform to get the denoised image back
        f_ishifted = torch.fft.ifftshift(f_shifted_filtered)
        img_back = torch.fft.ifft2(f_ishifted)
        img_back = torch.abs(img_back)
        return img_back
