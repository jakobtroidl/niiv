import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

class SSIM_Loss(torch.nn.Module):
    def __init__(self, range=1.0):
        super(SSIM_Loss, self).__init__()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=range).to('cuda')

    def forward(self, prediction, gt):
        return 1.0 - self.ssim(prediction, gt)


def image_mse(mask, model_output, gt):
    model_output = model_output.squeeze()
    gt = gt.squeeze()
    if mask is None:
        return {'img_loss': ((model_output- gt) ** 2).mean()}
    else:
        return {'img_loss': (mask * (model_output - gt) ** 2).mean()}
    
def image_l1(model_output, gt):
    model_output = model_output.squeeze()
    gt = gt.squeeze()
    return torch.abs(model_output - gt).mean()


def charbonnier_loss(prediction, gt, epsilon=1e-6):
    diff = prediction - gt
    loss = torch.sum(torch.sqrt(diff * diff + epsilon))
    return loss


def multi_frame_loss(model_output, gt, overlap, overlap_gt):

    weight_img_loss = 1.0
    weight_overlap_loss = 0.0
    
    image_loss = image_l1(model_output, gt)
    overlap_loss = image_l1(overlap, overlap_gt)

    total_loss = weight_img_loss * image_loss + weight_overlap_loss * overlap_loss
    
    return [total_loss, image_loss, overlap_loss]
