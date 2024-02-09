import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.image import image_gradients

class GradientRegularizer(nn.Module):
    def __init__(self):
        super(GradientRegularizer, self).__init__()

    def forward(self, x):
        # Compute the gradient of the image
        # and returns the sum of the gradient magnitudes normalized by the image size
        x_dim, y_dim = x.shape[-2:]
        x = x.unsqueeze(1)
        dy, dx = image_gradients(x)
        mag = torch.sqrt(dy**2 + dx**2)
        sum = torch.sum(mag, dim=(-2, -1))
        avg_sum = torch.mean(sum.squeeze())
        return avg_sum

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
