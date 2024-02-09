import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientRegularizer(nn.Module):
    def __init__(self):
        super(GradientRegularizer, self).__init__()
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).cuda()
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).cuda()

    def forward(self, x):
        x = x.unsqueeze(1)
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)


        print(grad_x.min(), grad_x.max())
        print(grad_y.min(), grad_y.max())


        abs = (torch.abs(grad_x) + torch.abs(grad_y)) / 2.0
        mean = torch.mean(abs)
        return mean
