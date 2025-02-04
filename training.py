import torch
import niiv.util.utils as utils
from tqdm.autonotebook import tqdm
import numpy as np
import os
from ignite.metrics import PSNR, SSIM

import math
import niiv.regularizer as regularizer
import wandb
from DISTS_pytorch import DISTS
import lpips

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
    lpips_loss = lpips.LPIPS(net='vgg').cuda() # closer to "traditional" perceptual loss, when used for optimization

    model.train()

    for epoch in tqdm(range(epochs)):
        if not epoch % epochs_til_checkpoint and epoch:
            torch.save({"model": model.state_dict()},
                           os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
            np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                           np.array(train_losses))
            
        for step, data in enumerate(train_dataloader):
            xy_output = model(data["xy"][0], data["xy"][1]) # inference on simulated xy anisotropic slice
            xy_gt = data["xy"][2].squeeze(1) # ground truth xy anisotropic slice

            im_size = int(math.sqrt(xy_output.shape[-2]))
            xy_output = xy_output.view(-1, *(im_size, im_size))

            mse = mse_loss(xy_output, xy_gt)

            # grad_reg = gradient_regularizer(xy_output, epoch, step, weight=1.0)
            # epoch_weight = torch.sigmoid(0.5 * torch.tensor(epoch) - 70).item() # weight the gradient regularizer less at the beginning of training

            xy_out_normed =  (xy_output.unsqueeze(1) * 2 - 1)
            xy_gt_normed = (xy_gt.unsqueeze(1) * 2 - 1)

            xy_out_rgb = xy_out_normed.expand(-1, 3, -1, -1)
            xy_gt_rgb = xy_gt_normed.expand(-1, 3, -1, -1)

            perc_loss = lpips_loss.forward(xy_out_rgb, xy_gt_rgb).mean()

            dists_loss = D(xy_output.unsqueeze(1), xy_gt.unsqueeze(1), require_grad=True, batch_average=True) 
            mae = mae_loss(xy_output, xy_gt)
            total_loss = 30 * mae + dists_loss
            total_loss_avg.add(total_loss.item())

            psnr_metric.update((xy_output, xy_gt))
            psnr = psnr_metric.compute()
            psnr_metric.reset()

            optim.zero_grad()
            total_loss.backward()
            optim.step()

            wandb.log({ "Total Loss": total_loss_avg.item(), 
                        "MSE Loss": mse.item(),
                        "DISTS Loss": dists_loss.item(),
                        "Train PSNR": psnr, 
                    })

            if not total_steps % steps_til_summary:
                tqdm.write("Epoch {}, Total Loss {}, MAE Loss {}, DISTS Loss {}".format(epoch, total_loss.item(), mae.item(), dists_loss.item()))

                torch.save({'epoch': total_steps,
                                    'model': model.state_dict(),
                                    'optimizer': optim.state_dict(),
                                    'scheduler': scheduler.state_dict(),
                                    }, os.path.join(checkpoints_dir, 'model_latest.pth'))

            total_steps += 1
            # train_dataloader.dataset.shuffle_x_or_y()

        scheduler.step()

    wandb.finish()

    torch.save({'epoch': total_steps,
                'model': model.state_dict(),
                'optimizer': optim.state_dict(),
                'scheduler': scheduler.state_dict(),  
                }, os.path.join(checkpoints_dir, f'model_final.pth'))
        
    return psnr