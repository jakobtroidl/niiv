import torch
from torch.fft import fft2, fftshift
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

def write_metrics_string(metrics, names):
    out = ""
    for i, name in enumerate(names):
        out += f"{name}: {metrics[:, i].mean().item():.2f}, "
    return out


def compute_all_metrics(pred, gt, im_max=1.0, low_pass_radius=25, im_size=(128, 128)):
    image_psnr = compute_metric(ImagePSNR(), pred, gt)
    image_ssim = compute_metric(ImageSSIM(), pred, gt)
    clipped_fourier_psnr = compute_metric(ClippedFourierPSNR(im_max, low_pass_radius, im_size), pred, gt)

    return torch.stack((image_psnr, image_ssim, clipped_fourier_psnr), dim=1)


def compute_metric(metric, pred, gt):
    return metric(pred, gt)

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
        sq_error = torch.abs(pred_fourier - gt_fourier) ** 2
        sum_sq_error = torch.sum(sq_error, dim=[-3, -2, -1])
        cf_psnr = 20 * torch.log10(self.im_max * n_pixels) - 10 * torch.log10(sum_sq_error)
        return cf_psnr
    
class ImagePSNR(torch.nn.Module):
    def __init__(self, im_max=1.0):
        super(ImagePSNR, self).__init__()
        self.psnr_metric = PeakSignalNoiseRatio(data_range=im_max, reduction='none', dim=[1, 2, 3]).cuda()

    def forward(self, prediction, gt):
        return self.psnr_metric(prediction, gt)


class ImageSSIM(torch.nn.Module):
    def __init__(self, im_max=1.0):
        super(ImageSSIM, self).__init__()
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=im_max, reduction='none').cuda()

    def forward(self, prediction, gt):
        return self.ssim_metric(prediction, gt)