import torch
import util.utils as utils
from tqdm.autonotebook import tqdm
import numpy as np
import os
from util.loss_functions import multi_frame_loss
from ignite.metrics import PSNR
import math

import wandb

def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, summary_fn, opt):

    run = wandb.init(project="continuous-volumes", group=opt.experiment_name, config=opt)

    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[2000, 4000, 6000, 8000], gamma=0.5)

    train_loss = utils.Averager()

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    total_steps = 0
    train_losses = []
    psnr_metric = PSNR(data_range=1.0)  # Use data_range=255 for images in [0, 255]


    for epoch in tqdm(range(epochs)):
        if not epoch % epochs_til_checkpoint and epoch:
            torch.save({"model": model.state_dict()},
                           os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
            np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                           np.array(train_losses))
            
        for step, data in enumerate(train_dataloader):

            xy_output = model(data["xy"][0], data["xy"][1]) # inference on simulated xy anisotropic slice
            slice_output = model(data["slice"][0], data["slice"][1]) # inference on simulated xz or yz anisotropic slice

            im_size = int(math.sqrt(xy_output.shape[-2]))

            overlap_axis = data["meta"][0] # 0 for x, 1 for y, both above outputs overlap either in x or y axis
            xy_overlap_idx = data["meta"][1].cuda() # z_depth of xy slice
            
            xy_overlap_idx = xy_overlap_idx.unsqueeze(-1)

            slice_overlap_idx = data["meta"][2].cuda() # x or y depth of xz or yz slice
            slice_overlap_idx = slice_overlap_idx.unsqueeze(-1)
            # slice_overlap_idx = slice_overlap_idx.view(-1, 1).expand(-1, im_size)

            # reformat to [batch, x, y]
            xy_output = xy_output.view(-1, *(im_size, im_size))
            slice_output = slice_output.view(-1, *(im_size, im_size))
            batch_indices = torch.arange(xy_output.size(0)).unsqueeze(-1)
            row_column_indices = torch.arange(im_size)

            if overlap_axis.eq(1).all(): # if overlap is in y axis
                xy_overlap = xy_output[batch_indices, slice_overlap_idx, row_column_indices]
                slice_overlap = slice_output[batch_indices, row_column_indices, xy_overlap_idx]
            elif overlap_axis.eq(0).all(): # if overlap is in x axis
                xy_overlap = xy_output[batch_indices, row_column_indices, slice_overlap_idx]
                slice_overlap = slice_output[batch_indices, row_column_indices, xy_overlap_idx]
            
            xy_gt = data["xy"][2].squeeze(1) # ground truth xy anisotropic slice

            # xy reconstruction needs to be accurate and overlap needs to be accurate
            [total_loss, loss, overlap] = multi_frame_loss(xy_output, xy_gt, xy_overlap, slice_overlap)
            train_loss.add(total_loss.item())

            psnr_metric.update((xy_output, xy_gt))
            psnr = psnr_metric.compute()
            psnr_metric.reset()

            wandb.log({"train_loss": loss.item(), "train_psnr": psnr, "overlap_loss": overlap.item()})
            
            optim.zero_grad()
            total_loss.backward()
            optim.step()

            if not total_steps % steps_til_summary:
                tqdm.write("Epoch {}, Total loss {}, PSNR {}".format(epoch, train_loss.item(), psnr))

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

