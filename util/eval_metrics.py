import torch
from torch.fft import fft2, fftshift, ifft2, ifftshift
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

def write_metrics_string(metrics, names):
    out = ""
    for i, name in enumerate(names):
        out += f"{name}: {metrics[:, i].mean().item():.2f}, "
    return out


def compute_all_metrics(pred, gt, im_max=1.0, im_size=128):
    image_psnr = compute_metric(ImagePSNR(), pred, gt)
    image_ssim = compute_metric(ImageSSIM(), pred, gt)
    clipped_fourier_psnr = compute_metric(ClippedFourierPSNR(im_max, im_size), pred, gt)

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

        delta = radius // 2
        mid = self.cols // 2
        start = max(mid - delta, 0)
        end = min(mid + delta, self.cols)
        # low_pass[:, start:end ] = 1
        low_pass[mask_area] = 1

        return low_pass.to("cuda")

class MultiClippedFourierPSNR(torch.nn.Module):
    def __init__(self, n_thresholds=64, im_max=1.0, im_size=128) -> None:
        super(MultiClippedFourierPSNR, self).__init__()
        self.cf_psnr = ClippedFourierPSNR(im_max, im_size)
        end = (im_size * torch.sqrt(torch.tensor(2))) // 2
        # end = 128
        t = torch.linspace(2, end, n_thresholds)
        self.thresholds = torch.round(t).int()

    def thresholds(self):
        return self.thresholds.tolist()
    
    def forward(self, prediction, gt):
        cf_psnr = torch.zeros((prediction.shape[0], self.thresholds.shape[0]), dtype=torch.float32).to("cuda")
        for i, threshold in enumerate(self.thresholds):
            cf_psnr[:, i] = self.cf_psnr(prediction, gt, threshold)
        return cf_psnr

class FourierDenoiser(torch.nn.Module):
    def __init__(self, threshold=25):
        super(FourierDenoiser, self).__init__()
        self.filter_threshold = threshold
    
    def forward(self, x):
        f_transform = fft2(x)
        f_shifted = fftshift(f_transform)
        low_pass = LowPassFilter(x.shape[-2], x.shape[-1])(self.filter_threshold)
        f_filtered = f_shifted * low_pass
        return torch.abs(ifft2(ifftshift(f_filtered)))


class ClippedFourierPSNR(torch.nn.Module):
    def __init__(self, im_max=1.0, im_size=128):
        super(ClippedFourierPSNR, self).__init__()
        self.im_max = torch.tensor(im_max)
        self.low_pass = LowPassFilter(im_size, im_size)

    def forward(self, prediction, gt, threshold=25):
        n_pixels = prediction.shape[-1] * prediction.shape[-2]
        pred_fourier = fftshift(fft2(prediction)) * self.low_pass(threshold)
        gt_fourier = fftshift(fft2(gt)) * self.low_pass(threshold)
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