import torch
import configargparse
import json
import os
import time

from torch.utils.data import DataLoader
from benchmarks.siren.data import SIRENData
from benchmarks.siren.field import FieldSiren
from niiv.util.eval_metrics import compute_all_metrics, write_metrics_string
from niiv.util.utils import exclude_max_min
from dataio import create_dir, save_images
import numpy as np

def test(opt):

    with open(opt.config, 'r') as f:
        config = json.load(f)

    metric_names = ["PSNR", "SSIM", "CF_PSNR"]
    log_dir = create_dir(opt.logging_root, opt.experiment_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    files = os.listdir(opt.dataset)

    info = os.path.join(os.path.dirname(os.path.dirname(opt.dataset)), "info.json")
    info = json.loads(open(info).read())
    has_isotropic_test_data = info["isotropic_test_data"]

    with torch.no_grad():
        result_metrics_all = torch.empty(0, len(metric_names)).cuda()
        times_all = []

        for file in files:
            data = os.path.join(opt.dataset, file)
            data_log_dir = os.path.join(log_dir, file)
            dataset = SIRENData(data, info)
            loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)


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

            result = torch.empty(dataset.gt_grid_size())
            t = 0

            for i, data in enumerate(loader):
                # gt_coords = dataset.sample_gt_coords(i, n_splits=batches)
                gt_coords = data
                shape = (gt_coords.shape[0], gt_coords.shape[1], gt_coords.shape[-1] * gt_coords.shape[-2] * gt_coords.shape[-3])
                gt_coords_linear = gt_coords.view(shape)
                gt_coords_linear = torch.permute(gt_coords_linear, (0, 2, 1))

                start = time.time()
                pred_values_linear = field(gt_coords_linear)
                t += time.time() - start

                pred_values = pred_values_linear.view(-1, data.shape[-3], data.shape[-2]).squeeze()
                result[:, :, i * opt.batch_size: (i + 1) * opt.batch_size] = torch.permute(pred_values, (1, 2, 0))

            times_all.append(t)
                

            if opt.render_as == "xz":
                pred_values_transformed = result.permute(1, 0, 2)
            elif opt.render_as == "yz":
                pred_values_transformed = result
            elif opt.render_as == "xy":
                pred_values_transformed = result.permute(2, 0, 1)
            else:
                raise ValueError("render_as must be one of 'xy', 'xz', 'yz'")
                
            pred_values_transformed = pred_values_transformed.unsqueeze(1)
            result = result.unsqueeze(1)

            output = "-------------------------------\n"
            output += "Sequence: {}\n".format(file)

            if has_isotropic_test_data:
                gt_values = dataset.sample_gt_values()
                gt_values = gt_values.squeeze().unsqueeze(1)

                metrics = compute_all_metrics(result.cuda(), gt_values)
                result_metrics_all = torch.cat((result_metrics_all, metrics), dim=0)
                metric_string = write_metrics_string(metrics, metric_names)
                output += "Avg Result Metrics: {}\n".format(metric_string)
                save_images(gt_dir, gt_values)
                save_images(res_dir, pred_values_transformed, metrics=metrics, metric_idx=0)
            else:
                save_images(res_dir, pred_values_transformed)

            output += "Time: {} sec\n".format(t)
            print(output)

            with open(os.path.join(data_log_dir, "iteration_{}".format(opt.iteration), "results.txt"), "w") as f:
                f.write(output)
    
    output = "-------------------------------\n"
    output += "Summary Results\n"
    if has_isotropic_test_data:
        result_metric_string = write_metrics_string(result_metrics_all, metric_names)
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
    p.add_argument('--batch_size', type=int, default=40, help="Batch size")
    p.add_argument('--iteration', type=int, help="i-th iteration of model evaluation. Default is 0.", default=0)
    p.add_argument('--render_as', type=str, help="Render the volume as xy, xz, or yz. Default is xz.", default="xz")
    opt = p.parse_args()

    test(opt)