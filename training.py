import torch
import niiv.util.utils as utils
from tqdm.autonotebook import tqdm
import numpy as np
import os
from ignite.metrics import PSNR, SSIM
import torch.nn.functional as F


import math
import niiv.regularizer as regularizer
import wandb
from DISTS_pytorch import DISTS
import lpips
import torchvision.transforms.functional as TF
import math

import torchvision.utils as vutils



def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, summary_fn, opt):

    run = wandb.init(project="continuous-volumes", group=opt.experiment_name, config=opt)

    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[2000, 4000, 6000, 8000], gamma=0.5)

    total_loss_avg = utils.Averager()
    # avg_pool = torch.nn.AvgPool2d(kernel_size=[1, int(7.5)]) 
    gradient_regularizer = regularizer.GradientRegularizer()

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)


    total_steps = 0
    train_losses = []
    psnr_metric = PSNR(data_range=1.0)  # Use data_range=255 for images in [0, 255]

    mse_loss = torch.nn.MSELoss()
    mae_loss = torch.nn.L1Loss()
    D = DISTS().cuda()

    feat_reg = regularizer.FeatureRegularizer()

    lpips_loss = lpips.LPIPS(net='vgg').cuda() # closer to "traditional" perceptual loss, when used for optimization

    model.train()

    for epoch in tqdm(range(epochs)):
        if not epoch % epochs_til_checkpoint and epoch:
            torch.save({"model": model.state_dict()},
                           os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
            np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                           np.array(train_losses))
            
        for step, data in enumerate(train_dataloader):
            x_degraded_out, x_deg_features = model(data["x_degraded"][0], data["x_degraded"][1]) # inference on simulated xy anisotropic slice
            y_degraded_out, y_deg_features = model(data["y_degraded"][0], data["y_degraded"][1]) # inference on simulated yz anisotropic slice

            B, N, C = x_degraded_out.shape
            im_size = int(math.sqrt(N))

            debug_x = x_degraded_out.view(-1, im_size, im_size, 1)
            image = TF.to_pil_image(debug_x[0, ...].squeeze(-1))
            image.save("x_degraded_out.png")

            y_degraded_out = y_degraded_out.view(-1, im_size, im_size, 1)
            y_degraded_out = y_degraded_out.permute(0, 2, 1, 3)

            image = TF.to_pil_image(y_degraded_out[0, ...].squeeze(-1))
            image.save("y_degraded_out.png")

            y_degraded_out = y_degraded_out.reshape(B, N, C)

            y_deg_features = y_deg_features.view(-1, im_size, im_size, y_deg_features.shape[-1])
            y_deg_features = y_deg_features.permute(0, 2, 1, 3)
            y_deg_features = y_deg_features.reshape(B, N, y_deg_features.shape[-1])

            reg, dists = feat_reg(y_deg_features[..., :64], x_deg_features[..., :64])
            # reg, dists = feat_reg(y_degraded_out, x_degraded_out)

            dists_out = dists.view(-1, im_size, im_size)
            # normalize to [0, 1]
            dists_out = (dists_out - dists_out.min()) / (dists_out.max() - dists_out.min())

            vutils.save_image(dists_out[0, ...], "output.png")

            xy_output = (x_degraded_out + y_degraded_out) / 2 

            # xy_output_plot = xy_output.view(-1, im_size, im_size, 1)
            # image = TF.to_pil_image(xy_output_plot[0, ...].squeeze(-1))
            # image.save("x_degraded_y_degraded_combinbed_out.png")
            # xy_output = x_degraded_out

            xy_gt = data["x_degraded"][2].squeeze(1)
            # image = TF.to_pil_image(xy_gt[0, ...])
            # image.save("xy_gt_out.png")   # ground truth xy anisotropic slice

            im_size = int(math.sqrt(xy_output.shape[-2]))
            xy_output = xy_output.view(-1, *(im_size, im_size))
            y_degraded_out = y_degraded_out.view(-1, *(im_size, im_size))
            x_degraded_out = x_degraded_out.view(-1, *(im_size, im_size))

            mse = mse_loss(y_degraded_out, xy_gt) + mse_loss(x_degraded_out, xy_gt)
            dists_loss = D(xy_output.unsqueeze(1), xy_gt.unsqueeze(1), require_grad=True, batch_average=True) 
            
            mae_x = mae_loss(x_degraded_out, xy_gt)
            mae_y = mae_loss(y_degraded_out, xy_gt)
            mae = mae_x + mae_y 

            # mae = mae_loss(xy_output, xy_gt)
            total_loss = mae # + reg
            total_loss_avg.add(total_loss.item())

            psnr_metric.update((xy_output, xy_gt))
            psnr = psnr_metric.compute()
            psnr_metric.reset()

            psnr_metric.update((x_degraded_out, xy_gt))
            psnr_x = psnr_metric.compute()
            psnr_metric.reset()

            psnr_metric.update((y_degraded_out, xy_gt))
            psnr_y = psnr_metric.compute()
            psnr_metric.reset()

            optim.zero_grad()
            total_loss.backward()
            optim.step()

            wandb.log({ "Total Loss": total_loss_avg.item(), 
                        "MSE Loss": mse.item(),
                        "DISTS Loss": dists_loss.item(),
                        "Train PSNR": psnr, 
                        "Train PSNR X": psnr_x,
                        "Train PSNR Y": psnr_y,
                        "Feat Dists": reg.item(),
                    })

            if not total_steps % steps_til_summary:
                tqdm.write("Epoch {}, Total Loss {}, MAE Loss {}, DISTS Loss {}".format(epoch, total_loss.item(), mae.item(), dists_loss.item()))

                torch.save({'epoch': total_steps,
                                    'model': model.state_dict(),
                                    'optimizer': optim.state_dict(),
                                    'scheduler': scheduler.state_dict(),
                                    }, os.path.join(checkpoints_dir, 'model_latest.pth'))

            total_steps += 1
        scheduler.step()

    wandb.finish()
    torch.save({'epoch': total_steps,
                'model': model.state_dict(),
                'optimizer': optim.state_dict(),
                'scheduler': scheduler.state_dict(),  
                }, os.path.join(checkpoints_dir, f'model_final.pth'))
        
    return psnr