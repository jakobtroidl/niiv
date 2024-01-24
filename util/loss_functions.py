import torch

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


def multi_frame_loss(model_output, gt, overlap, overlap_gt):

    weight_img_loss = 1.0
    weight_overlap_loss = 0.0
    
    image_loss = image_l1(model_output, gt)
    overlap_loss = image_l1(overlap, overlap_gt)

    total_loss = weight_img_loss * image_loss + weight_overlap_loss * overlap_loss
    
    return [total_loss, image_loss, overlap_loss]
