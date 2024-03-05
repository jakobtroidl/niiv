import torch
import niiv.util.utils as utils
from tqdm.autonotebook import tqdm
import numpy as np
import os
# from util.loss_functions import SSIM_Loss
from ignite.metrics import PSNR, SSIM

import math
import niiv.regularizer as regularizer
import wandb
from DISTS_pytorch import DISTS
# from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
import lpips
# from util.adists import ADISTS, prepare_image



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
    # ssim_metric = SSIM(data_range=1.0)  # Use data_range=255 for images in [0, 255]

    # ssim_loss = SSIM_Loss(range=1.0)
    mse_loss = torch.nn.MSELoss()
    mae_loss = torch.nn.L1Loss()
    D = DISTS().cuda()
    # adists = ADISTS().cuda()

    # ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0, kernel_size=9, betas=(0.2856, 0.3001, 0.2363, 0.1333)).cuda()
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

            # slice_input = data["slice"][0]
            # slice_coords = data["slice"][1]
            # slice_output = model(slice_input, slice_coords) # inference on simulated xz or yz anisotropic slice      
            # slice_output = slice_output.view(-1, *(im_size, im_size)).unsqueeze(1)
            # slice_output = avg_pool(slice_output)

            # xy_loss = charbonnier_loss(xy_output, xy_gt)
            # slice_loss = charbonnier_loss(slice_input, slice_output)

            # xy_loss_ssim = ssim_loss(xy_output.unsqueeze(1), xy_gt.unsqueeze(1))
            mse = mse_loss(xy_output, xy_gt)
            # weighted_xy_mse_loss = 55 * xy_loss_mse

            # # plot_histogram = epoch % 5 == 0 and step == 0 
            grad_reg = gradient_regularizer(xy_output, epoch, step, weight=1.0)

            epoch_weight = torch.sigmoid(0.5 * torch.tensor(epoch) - 70).item() # weight the gradient regularizer less at the beginning of training
            # # grad_reg = epoch_weight * grad_reg

            xy_out_normed =  (xy_output.unsqueeze(1) * 2 - 1)
            xy_gt_normed = (xy_gt.unsqueeze(1) * 2 - 1)

            xy_out_rgb = xy_out_normed.expand(-1, 3, -1, -1)
            xy_gt_rgb = xy_gt_normed.expand(-1, 3, -1, -1)

            perc_loss = lpips_loss.forward(xy_out_rgb, xy_gt_rgb).mean()

            # weighted_reg = 1 * grad_reg
            dists_loss = D(xy_output.unsqueeze(1), xy_gt.unsqueeze(1), require_grad=True, batch_average=True) 
            # ms_loss = 1.0 - ms_ssim(xy_output.unsqueeze(1), xy_gt.unsqueeze(1))
            mae = mae_loss(xy_output, xy_gt)

            # print(f"dists_loss: {dists_loss}, mae_weighted: {mae_weighted}")

            # total_loss = xy_loss_ssim + xy_loss_mse + weighted_reg
            # total_loss = weighted_xy_mse_loss + weighted_reg
            # total_loss = dists_loss + 9 * mae
            # total_loss = 20 * mae + dists_loss # + 3 * grad_reg

            # ad_loss = adists(xy_out_rgb, xy_gt_rgb, as_loss=True)

            # total_loss = 30 * mae + dists_loss
            total_loss = 30 * mae
            total_loss_avg.add(total_loss.item())

            psnr_metric.update((xy_output, xy_gt))
            psnr = psnr_metric.compute()
            psnr_metric.reset()

            # ssim_metric.update((xy_output.unsqueeze(1), xy_gt.unsqueeze(1)))
            # ssim = ssim_metric.compute()
            # ssim_metric.reset()

            optim.zero_grad()
            total_loss.backward()
            optim.step()

            wandb.log({ "Total Loss": total_loss_avg.item(), 
                        "MSE Loss": mse.item(),
                        "DISTS Loss": dists_loss.item(),
                        "Grad Reg": grad_reg.item(),
                        "Epoch Weight": epoch_weight,
                    #    "XY Loss MSE": xy_loss_mse.item(), 
                    #     "XY Loss SSIM": xy_loss_ssim.item(),
                    #    # "Overlap Loss": slice_loss.item(), 
                    #    "Edge Loss": grad_reg.item(),
                        "Train PSNR": psnr, 
                    #    "train_ssim": ssim,
                    })

            if not total_steps % steps_til_summary:
                # tqdm.write("Epoch {}, Total Loss {}, XY Loss MSE {}, XY Loss SSIM {}, Grad Regularizer Term {}, PSNR {}, ".format(epoch, total_loss.item(), xy_loss_mse, xy_loss_ssim, grad_reg, psnr))
                # tqdm.write("Epoch {}, Total Loss {}, DISTS Loss {}, MAE Loss {}, Grad Reg {}, PSNR {}, ".format(epoch, total_loss.item(), dists_loss.item(), mae.item(), grad_reg.item(), psnr))
                tqdm.write("Epoch {}, Total Loss {}, MAE Loss {}, DISTS Loss {}, Epoch Weight {}".format(epoch, total_loss.item(), mae.item(), dists_loss.item(), epoch_weight))

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