import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.image import image_gradients
from niiv.util.transforms import crop_image_border
import matplotlib.pyplot as plt
from niiv.util.eval_metrics import FourierDenoiser
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import torchvision.utils as vutils



class FeatSpaceSmoothness(nn.Module):
    def __init__(self):
        super(FeatSpaceSmoothness, self).__init__()
        self.n_samples = 500
    
    def forward(self, z, y):
        """
        Enforces similar latents for similar outputs.

        Args:
        z: Tensor of shape (batch_size, latent_dim) - Latent representations
        y: Tensor of shape (batch_size,) - Continuous target values

        Returns:
        loss: Smoothness loss
        """

        z = z.view(-1, z.shape[-1])
        y = y.unsqueeze(-1).view(-1, 1)

        idx = random.sample(range(z.shape[0]), self.n_samples)

        z = z[idx]
        y = y[idx]

        feat_diff = z.unsqueeze(1) - z.unsqueeze(0)
        out_diff = y.unsqueeze(1) - y.unsqueeze(0)

        pairwise_distances = torch.norm(feat_diff, p=2, dim=-1)  # Euclidean distance between latents
        output_distances = torch.abs(out_diff.squeeze(-1)) * 100   # Distance between outputs

        # vutils.save_image(pairwise_distances / pairwise_distances.max(), "latent_distances.png")
        # vutils.save_image(output_distances / output_distances.max(), "output_distances.png")
        
        loss = (pairwise_distances * output_distances).mean()
        return loss


class FeatPairSimilarity(nn.Module):

    def __init__(self):
        super(FeatPairSimilarity, self).__init__()
        self.n_samples = 5000
    
    def forward(self, features, outputs, similarity_metric="euclidean"):
        """
        Enforce that similar features have similar outputs.
        
        Args:
            features (torch.Tensor): Input features of shape (N, D), where N is the batch size.
            outputs (torch.Tensor): Corresponding model outputs of shape (N, O), where O is output size.
            similarity_metric (str): "euclidean" or "cosine".
        
        Returns:
            torch.Tensor: Regularization loss.
        """

        B, N, C = features.shape

        features = features.view(-1, C)
        outputs = outputs.view(-1, 1)

        # Randomly sample a subset of features
        idx = random.sample(range(features.shape[0]), self.n_samples)

        features = features[idx]
        outputs = outputs[idx]



        # Compute pairwise distance between features
        if similarity_metric == "euclidean":
            feature_dist = torch.cdist(features, features, p=2)  # Pairwise L2 distance
        elif similarity_metric == "cosine":
            feature_dist = 1 - F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)
        else:
            raise ValueError("Unsupported similarity metric")

        # Compute pairwise distance between outputs
        output_dist = torch.cdist(outputs, outputs, p=2)

        # Regularization loss: minimize difference in output distances for similar features
        loss = F.mse_loss(output_dist, feature_dist)

        return loss


class FeatureRegularizer(nn.Module):
    def __init__(self):
        super(FeatureRegularizer, self).__init__()

    def forward(self, f1, f2):
        # f1 = F.normalize(f1, p=2, dim=-1)
        # f2 = F.normalize(f2, p=2, dim=-1)

        # dists = torch.norm(f1 - f2, p=2, dim=-1) 
        # # dists = dists / 2.0 # normalize to [0, 1]
        # #dists = torch.abs(dists)
        # avg = torch.mean(dists)
        # #avg = torch.mean(avg)

        dists = 1.0 - F.cosine_similarity(f1, f2, dim=-1)
        avg = torch.mean(dists)


        plt.hist(dists.flatten().detach().cpu().numpy(), bins=100)
        plt.savefig('dists-histogram.png')
        plt.clf()

        return avg, dists


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
