import torch
import os

from pytorch_msssim import ms_ssim
class Averager():
    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v

def msssim_fn(output, target):
    if output.size(-2) >= 160:
        msssim = ms_ssim(output.float().detach(), target.detach(), data_range=1, size_average=True)
    else:
        msssim = torch.tensor(0).to(output.device)

    return msssim

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
    
def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def write_volume_summary(volume_dataset, model, model_input, gt, writer, total_steps, prefix='train_'):
    print("write_volume_summary")
    PSNR = 0.0 # TODO
    return PSNR

def min_max_summary(name, tensor, writer, total_steps):
    writer.add_scalar(name + '_min', tensor.min().detach().cpu().numpy(), total_steps)
    writer.add_scalar(name + '_max', tensor.max().detach().cpu().numpy(), total_steps)

# from LIIF code base (Chen. 2021)
def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        # seq = v0 + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def exclude_max_min(list):
    sorted_list = sorted(list)
    return sorted_list[1:-1]





