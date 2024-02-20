from pathlib import Path

import hydra
import torch
from jaxtyping import install_import_hook
from omegaconf import DictConfig
from torch import nn
from tqdm import trange
import configargparse


from src.dataset import get_field_dataset
from src.field import get_field
from src.sampling import generate_random_samples, sample_grid
from src.visualization.image import save_rendered_field



def train(opt):
    # Set up the dataset, field, optimizer, and loss function.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = get_field_dataset(opt)
    field = get_field(opt, dataset.d_coordinate, dataset.d_out).to(device)
    optimizer = torch.optim.Adam(
        field.parameters(),
        lr=opt.learning_rate,
    )
    loss_fn = nn.MSELoss()

    # Optionally re-map the outputs to the neural field so they're in range [0, 1].
    if opt.remap_outputs:
        field = nn.Sequential(
            field,
            nn.Sigmoid(),
        )

    # Fit the field to the dataset.
    for iteration in (progress := trange(opt.num_iterations)):
        optimizer.zero_grad()
        samples = generate_random_samples(dataset.d_coordinate, opt.batch_size, device)
        predicted = field(samples)
        ground_truth = dataset.query(samples)
        loss = loss_fn(predicted, ground_truth)
        loss.backward()
        optimizer.step()

        # Intermittently visualize training progress.
        if iteration % opt.visualization_interval == 0:
            with torch.no_grad():
                # Render the field in a grid.
                samples = sample_grid(dataset.grid_size, device)
                *dimensions, d_coordinate = samples.shape
                samples = samples.reshape(-1, d_coordinate)
                values = [field(batch) for batch in samples.split(opt.batch_size)]
                values = torch.cat(values).reshape(*dimensions, -1)

                # Save the result.
                path = Path(f"{opt.output_path}/{iteration:0>6}.png")
                save_rendered_field(values, path)

        progress.desc = f"Training (loss: {loss.item():.4f})"


if __name__ == "__main__":

    p = configargparse.ArgumentParser()
    p.add('-c', '--config', required=True,  help='Path to config file. (e.g., ./config/config_small.json)')
    p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
    p.add_argument('--experiment_name', type=str, required=False, default="",
                help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

    # General training options
    p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-2')
    p.add_argument('--num_epochs', type=int, default=200, help='Number of epochs to train for.')
    p.add_argument('--epochs_til_ckpt', type=int, default=1000, help='Time interval in seconds until checkpoint is saved.')
    p.add_argument('--steps_til_summary', type=int, default=50,
                help='Time interval in seconds until tensorboard summary is saved.')
    p.add_argument('--dataset', type=str, required=True, help="Dataset Path, (e.g., /data/UVG/Jockey)")
    p.add_argument('--batch_size', type=int, default=6, help="Batch size")
    opt = p.parse_args()

    train(opt)