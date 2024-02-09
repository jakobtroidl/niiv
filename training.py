import torch
import util.utils as utils
from tqdm.autonotebook import tqdm
import numpy as np
import os
from util.loss_functions import image_l1, charbonnier_loss
from ignite.metrics import PSNR, SSIM
import math
import src.regularizer as regularizer
import wandb

def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, summary_fn, opt):

    run = wandb.init(project="continuous-volumes", group=opt.experiment_name, config=opt)

    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[2000, 4000, 6000, 8000], gamma=0.5)

    total_loss_avg = utils.Averager()
    avg_pool = torch.nn.AvgPool2d(kernel_size=[1, int(7.5)]) 
    gradient_regularizer = regularizer.GradientRegularizer()

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    total_steps = 0
    train_losses = []
    psnr_metric = PSNR(data_range=1.0)  # Use data_range=255 for images in [0, 255]
    ssim_metric = SSIM(data_range=1.0)  # Use data_range=255 for images in [0, 255]

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

            slice_input = data["slice"][0]
            slice_coords = data["slice"][1]
            slice_output = model(slice_input, slice_coords) # inference on simulated xz or yz anisotropic slice      
            slice_output = slice_output.view(-1, *(im_size, im_size)).unsqueeze(1)
            slice_output = avg_pool(slice_output)

            xy_loss = charbonnier_loss(xy_output, xy_gt)
            slice_loss = charbonnier_loss(slice_input, slice_output)
            edge_loss = gradient_regularizer(xy_output)

            # weight = min(1.0, ((1.0/30.0) * epoch) ** 2)
            weight = torch.sigmoid(torch.tensor(epoch) - 40)
            weight = torch.clamp(weight, min=1e-6)
            grad_reg = weight * edge_loss

            total_loss = (0.75 * xy_loss) + (0.25 * slice_loss) - grad_reg
            total_loss_avg.add(total_loss.item())

            psnr_metric.update((xy_output, xy_gt))
            psnr = psnr_metric.compute()
            psnr_metric.reset()

            ssim_metric.update((xy_output.unsqueeze(1), xy_gt.unsqueeze(1)))
            ssim = ssim_metric.compute()
            ssim_metric.reset()

            optim.zero_grad()
            total_loss.backward(retain_graph=True)
            optim.step()

            wandb.log({"Total Loss": total_loss_avg.item(), 
                       "XY Loss": xy_loss.item(), 
                       "Overlap Loss": slice_loss.item(), 
                       "Edge Loss": edge_loss.item(),
                       "train_psnr": psnr, 
                       "train_ssim": ssim,
                    })

            if not total_steps % steps_til_summary:
                tqdm.write("Epoch {}, Total Loss {}, PSNR {}, Weight Component {}.".format(epoch, total_loss.item(), psnr, grad_reg))

                torch.save({'epoch': total_steps,
                                    'model': model.state_dict(),
                                    'optimizer': optim.state_dict(),
                                    'scheduler': scheduler.state_dict(),
                                    }, os.path.join(checkpoints_dir, 'model_latest.pth'))

            total_steps += 1
            train_dataloader.dataset.shuffle_x_or_y()

        scheduler.step()

    wandb.finish()

    torch.save({'epoch': total_steps,
                'model': model.state_dict(),
                'optimizer': optim.state_dict(),
                'scheduler': scheduler.state_dict(),  
                }, os.path.join(checkpoints_dir, f'model_final.pth'))
        
    return psnr

