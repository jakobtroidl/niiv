import torch
from torch.fft import fft2, fftshift

class LowPassFilter(torch.nn.Module):
    def __init__(self, rows, cols) -> None:
        super(LowPassFilter, self).__init__()
        self.rows = rows
        self.cols = cols
    
    def forward(self, radius):
        crow, ccol = self.rows // 2 , self.cols // 2
        low_pass = torch.zeros((self.rows, self.cols), dtype=torch.uint8)
        x = torch.arange(0, self.cols).unsqueeze(0).expand(self.rows, self.cols)
        y = torch.arange(0, self.rows).unsqueeze(1).expand(self.rows, self.cols)
        mask_area = (x - ccol) ** 2 + (y - crow) ** 2 <= radius**2
        low_pass[mask_area] = 1
        return low_pass.to("cuda")

class ClippedFourierPSNR(torch.nn.Module):
    def __init__(self, im_max=1.0, low_pass_radius=25, im_size=(128, 128)):
        super(ClippedFourierPSNR, self).__init__()
        self.im_max = torch.tensor(im_max)
        self.threshold = low_pass_radius
        self.low_pass = LowPassFilter(im_size[0], im_size[1])

    def forward(self, prediction, gt):
        n_pixels = prediction.shape[-1] * prediction.shape[-2]
        pred_fourier = fftshift(fft2(prediction)) * self.low_pass(self.threshold)
        gt_fourier = fftshift(fft2(gt)) * self.low_pass(self.threshold)
        sum_sq_error = torch.sum(torch.abs(pred_fourier - gt_fourier) ** 2)
        cf_psnr = 20 * torch.log10(self.im_max * n_pixels) - 10 * torch.log10(sum_sq_error)
        return cf_psnr.item()