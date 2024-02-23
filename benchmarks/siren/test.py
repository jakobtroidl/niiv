import torch
import configargparse
import json
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

from benchmarks.siren.data import SIRENData
from benchmarks.siren.field import FieldSiren
from util.eval_metrics import ImagePSNR, ImageSSIM
from dataio import create_dir, save_images

def train(opt):

    with open(opt.config, 'r') as f:
        config = json.load(f)

    psnr = ImagePSNR()
    ssim = ImageSSIM()

    log_dir = create_dir(opt.logging_root, opt.experiment_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    files = os.listdir(opt.dataset)

    with torch.no_grad():

        for file in files:
            data = os.path.join(opt.dataset, file)
            data_log_dir = os.path.join(log_dir, file)
            dataset = SIRENData(data)

            # load the field from checkpoint
            model_path = os.path.join(data_log_dir, "model_latest.pth")
            # check if model path exists
            if not os.path.exists(model_path):
                print("Model {} does not exists. Skipping...".format(file))
                continue

            ckpt = torch.load(model_path)
            field = FieldSiren(config["siren"]).to(device)
            field.load_state_dict(ckpt['model'])
            field.eval()

            res_dir = create_dir(data_log_dir, "results_iteration_{}".format(opt.iteration))
            gt_coords, gt_values = dataset.sample_gt()
                    
            for i in tqdm(range(gt_coords.shape[-1])):
                im_size = 128
                z_coords = gt_coords[..., i].squeeze().view(-1, im_size**2).permute(1, 0)
                z_predicted = field(z_coords)
                z_predicted = z_predicted.view(im_size, im_size)
                psnr_val = torch.mean(psnr(z_predicted, gt_values[..., i]))
                im = z_predicted.squeeze().cpu().detach().numpy()
                im = im * 255
                im = im.astype(np.uint8)
                im = Image.fromarray(im)
                name = "siren_{}_psnr_{}.png".format(i, psnr_val.item())
                out = os.path.join(res_dir, name)
                im.save(out)

if __name__ == "__main__":

    p = configargparse.ArgumentParser()
    p.add('-c', '--config', required=True,  help='Path to config file. (e.g., ./config/config_small.json)')
    p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
    p.add_argument('--experiment_name', type=str, required=False, default="",
                help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

    # General training options
    p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-2')
    p.add_argument('--num_iterations', type=int, default=20000, help='Number of epochs to train for.')
    p.add_argument('--epochs_til_ckpt', type=int, default=1000, help='Time interval in seconds until checkpoint is saved.')
    p.add_argument('--steps_til_summary', type=int, default=50,
                help='Time interval in seconds until tensorboard summary is saved.')
    p.add_argument('--dataset', type=str, required=True, help="Dataset Path, (e.g., /data/UVG/Jockey)")
    p.add_argument('--batch_size', type=int, default=6, help="Batch size")
    p.add_argument('--iteration', type=int, help="i-th iteration of model evaluation. Default is 0.", default=0)
    opt = p.parse_args()

    train(opt)