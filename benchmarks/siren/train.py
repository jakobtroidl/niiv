import torch
from torch import nn
from tqdm import trange
import configargparse
import json
import os

from benchmarks.siren.data import SIRENData
from benchmarks.siren.field import FieldSiren
from util.eval_metrics import ImagePSNR
from dataio import create_dir

def train(opt):

    with open(opt.config, 'r') as f:
        config = json.load(f)

    psnr = ImagePSNR()
    log_dir = create_dir(opt.logging_root, opt.experiment_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.MSELoss()

    # list all files in the dataset directory
    files = os.listdir(opt.dataset)

    for file in files:
        path = os.path.join(opt.dataset, file)
        data_dir = create_dir(log_dir, file)
        dataset = SIRENData(path)
        field = FieldSiren(config["siren"]).to(device)
        optimizer = torch.optim.Adam(field.parameters(), lr=opt.lr)

        # Fit the field to the dataset.
        for iteration in (progress := trange(opt.num_iterations)):
            optimizer.zero_grad()
            samples, values = dataset.random_sample(opt.batch_size)
            predicted = field(samples)
            loss = loss_fn(predicted, values)
            loss.backward()
            optimizer.step()

            if iteration % opt.steps_til_summary == 0:
                # Log the loss to the progress bar.
                description = f"Training (loss: {loss.item():.4f})\n"
                pred_unsq = predicted.unsqueeze(-1).unsqueeze(-1)
                gt_unsq = values.unsqueeze(-1).unsqueeze(-1)
                description += f"PSNR: {torch.mean(psnr(pred_unsq, gt_unsq)).item():.4f}\n"
                progress.desc = description
            
            if iteration % opt.epochs_til_ckpt == 0:
                # Save the model checkpoint.
                torch.save(
                    {
                        "model": field.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(data_dir, "model_latest.pth"),
                )

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
    p.add_argument('--steps_til_summary', type=int, default=100,
                help='Time interval in seconds until tensorboard summary is saved.')
    p.add_argument('--dataset', type=str, required=True, help="Dataset Path, (e.g. path to folder of .npy volumes). ")
    p.add_argument('--batch_size', type=int, default=1024, help="Batch size")
    opt = p.parse_args()

    train(opt)