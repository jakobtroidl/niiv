import torch
import configargparse
import json
import os
import time

from benchmarks.siren.data import SIRENData
from benchmarks.siren.field import FieldSiren
from util.eval_metrics import compute_all_metrics, write_metrics_string
from util.utils import exclude_max_min
from dataio import create_dir, save_images
import numpy as np

def train(opt):

    with open(opt.config, 'r') as f:
        config = json.load(f)

    metric_names = ["PSNR", "SSIM", "CF_PSNR"]
    log_dir = create_dir(opt.logging_root, opt.experiment_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    files = os.listdir(opt.dataset)

    with torch.no_grad():
        result_metrics_all = torch.empty(0, len(metric_names)).cuda()
        times_all = []

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

            res_dir = create_dir(os.path.join(data_log_dir, "iteration_{}".format(opt.iteration)), "results")
            gt_dir = create_dir(os.path.join(data_log_dir, "iteration_{}".format(opt.iteration)), "gt")
            gt_coords, gt_values = dataset.sample_gt()

            gt_coords_linear = gt_coords.view(-1, gt_coords.shape[-1] * gt_coords.shape[-2] * gt_coords.shape[-3])
            gt_coords_linear = torch.permute(gt_coords_linear, (1, 0))

            start = time.time()
            pred_values_linear = field(gt_coords_linear)
            duration = time.time() - start

            times_all.append(duration)

            pred_values = pred_values_linear.view(gt_values.shape).squeeze()
            pred_values = pred_values.unsqueeze(1)
            gt_values = gt_values.squeeze().unsqueeze(1)

            metrics = compute_all_metrics(pred_values, gt_values)
            result_metrics_all = torch.cat((result_metrics_all, metrics), dim=0)
            metric_string = write_metrics_string(metrics, metric_names)

            output = "-------------------------------\n"
            output += "Sequence: {}\n".format(file)
            output += "Avg Result Metrics: {}\n".format(metric_string)
            output += "Time: {} sec\n".format(duration)
            print(output)

            with open(os.path.join(data_log_dir, "iteration_{}".format(opt.iteration), "results.txt"), "w") as f:
                f.write(output)
            
            save_images(res_dir, pred_values, metrics=metrics, metric_idx=0)
            save_images(gt_dir, gt_values)
    
    result_metric_string = write_metrics_string(result_metrics_all, metric_names)
    output = "-------------------------------\n"
    output += "Summary Results\n"
    output += "Avg Result Metrics: {}\n".format(result_metric_string)
    output += "Reconstruction Time: {} sec\n".format(np.mean(exclude_max_min(times_all)))

    print(output)
    with open(os.path.join(log_dir, "avg_result.txt"), "w") as f:
        f.write(output)

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