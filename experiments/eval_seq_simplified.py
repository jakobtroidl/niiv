# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio
import configargparse
import torch

import configargparse
from pynvml import *
import json
import numpy as np
from torch.utils.data import DataLoader 
import math
from torch.nn import functional as F
from niiv.models import NIIV
from niiv.util.eval_metrics import compute_all_metrics, write_metrics_string, MultiClippedFourierPSNR
from dataio import create_dir, save_images, save_ablation_linechart

import time

command_line = ""
for arg in sys.argv:
    command_line += arg
    command_line += " "

p = configargparse.ArgumentParser()
p.add('-c', '--config', required=True,  help='Path to config file.')
p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=False, default="", help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')
p.add_argument('--dataset', type=str, required=True, help="Dataset Path, (e.g., /data/UVG/Honeybee)")
p.add_argument('--iteration', type=int, required=False, default=0, help="i-th iteration of model evaluation")

opt = p.parse_args()

nvmlInit()
device_id = nvmlDeviceGetHandleByIndex(0)  # Assuming you're using the first GPU
pid = os.getpid()

dataset_name = opt.experiment_name

with open(opt.config, 'r') as f:
    config = json.load(f)

# Define the model.
model = NIIV(out_features=1, encoding_config=config["cvr"], export_features=False)
model.cuda()

config = config["cvr"]

# Model Loading
root_path = os.path.join(opt.logging_root, opt.experiment_name)

path = os.path.join(root_path, 'checkpoints', "model_epoch_0900.pth")
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model'])
model.eval()

dir = os.path.dirname(opt.dataset)
test_seq_dir = os.path.join(dir, "test")
results_dir = create_dir(root_path, 'results_iteration_{}'.format(opt.iteration))

metric_names = ["PSNR", "SSIM", "CF_PSNR"]
seq_names = os.listdir(test_seq_dir)
mcf_psnr = MultiClippedFourierPSNR()

result_metrics_all = torch.empty(0, len(metric_names)).cuda()
bilinear_metrics_all = torch.empty(0, len(metric_names)).cuda()
nearest_metrics_all = torch.empty(0, len(metric_names)).cuda()

result_mcf_psnr_all = torch.empty(0, mcf_psnr.thresholds.shape[0]).cuda()
bilinear_mcf_psnr_all = torch.empty(0, mcf_psnr.thresholds.shape[0]).cuda()
nearest_mcf_psnr_all = torch.empty(0, mcf_psnr.thresholds.shape[0]).cuda()

times_list = []
memory_list = []

for seq in seq_names:

    dataset = dataio.ImageDatasetTest(path_to_info=opt.dataset, name=seq)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=15, num_workers=0) # prev batch size 63

    seq_res_dir = create_dir(results_dir, seq)
    result_metrics = torch.empty(0, len(metric_names)).cuda()
    nearest_metrics = torch.empty(0, len(metric_names)).cuda()
    bilinear_metrics = torch.empty(0, len(metric_names)).cuda()

    result_mcf_psnr = torch.empty(0, mcf_psnr.thresholds.shape[0]).cuda()
    bilinear_mcf_psnr = torch.empty(0, mcf_psnr.thresholds.shape[0]).cuda()
    nearest_mcf_psnr = torch.empty(0, mcf_psnr.thresholds.shape[0]).cuda()
    
    times = []
    memory = []

    for step, (model_input, coords, gt, file_names) in enumerate(dataloader):

        with torch.no_grad():
            start = time.time()
            prediction = model(coords=coords, image=model_input)
            duration = time.time() - start
            times.append(duration)

            # Get the memory info for your specific process
            info = nvmlDeviceGetComputeRunningProcesses(device_id)

            for proc in info:
                if proc.pid == pid:
                    gpu_mem = proc.usedGpuMemory / 1024**2
                    memory.append(gpu_mem)
                    break

            result_dir = create_dir(seq_res_dir, "result")
            nearest_dir = create_dir(seq_res_dir, "nearest")
            bilinear_dir = create_dir(seq_res_dir, "bilinear")
            gt_dir = create_dir(seq_res_dir, "gt")
            input_dir = create_dir(seq_res_dir, "input")

            side_length = int(math.sqrt(prediction.shape[1]))
            pred_image = prediction.view(-1, side_length, side_length).unsqueeze(1)
            gt_image = gt.view(-1, side_length, side_length).unsqueeze(1)

            # export normal interpolated images
            nearest = F.interpolate(model_input, size=[side_length, side_length], mode='nearest')
            bilinear = F.interpolate(model_input, size=[side_length, side_length], mode='bilinear', align_corners=False)

            # compute PSNR and other metrics if isotropic test data is available 
            if dataset.has_isotropic_test_data():
                result_metrics = torch.cat((result_metrics, compute_all_metrics(pred_image, gt_image)), dim=0)
                nearest_metrics = torch.cat((nearest_metrics, compute_all_metrics(nearest, gt_image)), dim=0)
                bilinear_metrics = torch.cat((bilinear_metrics, compute_all_metrics(bilinear, gt_image)), dim=0)

                result_mcf_psnr = torch.cat((result_mcf_psnr, mcf_psnr(pred_image, gt_image)), dim=0)
                bilinear_mcf_psnr = torch.cat((bilinear_mcf_psnr, mcf_psnr(bilinear, gt_image)), dim=0)
                nearest_mcf_psnr = torch.cat((nearest_mcf_psnr, mcf_psnr(nearest, gt_image)), dim=0)

                save_images(result_dir, pred_image, file_names, metrics=result_metrics, metric_idx=0)
                save_images(nearest_dir, nearest, file_names, metrics=nearest_metrics, metric_idx=0)
                save_images(bilinear_dir, bilinear, file_names, metrics=bilinear_metrics, metric_idx=0)
                save_images(gt_dir, gt_image, file_names)
            else:
                save_images(result_dir, pred_image, file_names)
                save_images(nearest_dir, nearest, file_names)
                save_images(bilinear_dir, bilinear, file_names)

            save_images(input_dir, model_input, file_names)
    
    output = "-------------------------------\n"
    output += "Sequence: {}\n".format(seq)
    output += "Reconstruction Time: {} sec\n".format(np.sum(times))
    output += "Memory: {} MB\n".format(np.mean(memory))

    if dataset.has_isotropic_test_data():
        result_metric_string = write_metrics_string(result_metrics, metric_names)
        nearest_metric_string = write_metrics_string(nearest_metrics, metric_names)
        bilinear_metric_string = write_metrics_string(bilinear_metrics, metric_names)

        mean_result_mcf_psnr = torch.round(result_mcf_psnr.mean(dim=0), decimals=2)
        mean_bilinear_mcf_psnr = torch.round(bilinear_mcf_psnr.mean(dim=0), decimals=2)
        mean_nearest_mcf_psnr = torch.round(nearest_mcf_psnr.mean(dim=0), decimals=2)

        output += "Avg Result Metrics: {}\n".format(result_metric_string)
        output += "Avg Bilinear Metrics: {}\n".format(bilinear_metric_string)
        output += "Avg Nearest Metrics: {}\n".format(nearest_metric_string)

        out_mean_result_mcf_psnr = mean_result_mcf_psnr.detach().cpu().tolist()
        out_mean_bilinear_mcf_psnr = mean_bilinear_mcf_psnr.detach().cpu().tolist()
        out_mean_nearest_mcf_psnr = mean_nearest_mcf_psnr.detach().cpu().tolist()
        out_thresholds = mcf_psnr.thresholds.tolist()

        output += "-------------------------------\n"
        output += "Result MCF PSNR: {}\n".format(out_mean_result_mcf_psnr)
        output += "Bilinear MCF PSNR: {}\n".format(out_mean_bilinear_mcf_psnr)
        output += "Nearest MCF PSNR: {}\n".format(out_mean_nearest_mcf_psnr)
        output += "Thresholds: {}\n".format(out_thresholds)

        with open(os.path.join(seq_res_dir, "result.txt"), "w") as f:
            f.write(output)

        save_ablation_linechart(seq_res_dir, out_thresholds, out_mean_result_mcf_psnr, out_mean_bilinear_mcf_psnr, out_mean_nearest_mcf_psnr)

        result_metrics_all = torch.cat((result_metrics_all, result_metrics), dim=0)
        nearest_metrics_all = torch.cat((nearest_metrics_all, nearest_metrics), dim=0)
        bilinear_metrics_all = torch.cat((bilinear_metrics_all, bilinear_metrics), dim=0)


        result_mcf_psnr_all = torch.cat((result_mcf_psnr_all, mean_result_mcf_psnr.unsqueeze(0)), dim=0)
        bilinear_mcf_psnr_all = torch.cat((bilinear_mcf_psnr_all, mean_bilinear_mcf_psnr.unsqueeze(0)), dim=0)
        nearest_mcf_psnr_all = torch.cat((nearest_mcf_psnr_all, mean_nearest_mcf_psnr.unsqueeze(0)), dim=0)
    
    times_list.append(np.sum(times))
    memory_list.append(np.mean(memory))
    print(output)

output = "-------------------------------\n"
output += "Summary Results\n"
output += "Reconstruction Time: {} sec\n".format(np.mean(times_list))
output += "Memory: {} MB\n".format(np.mean(memory_list))


if dataset.has_isotropic_test_data():
    result_metric_string = write_metrics_string(result_metrics_all, metric_names)
    bilinear_metric_string = write_metrics_string(bilinear_metrics_all, metric_names)
    nearest_metric_string = write_metrics_string(nearest_metrics_all, metric_names)

    output += "Avg Result Metrics: {}\n".format(result_metric_string)
    output += "Avg Bilinear Metrics: {}\n".format(bilinear_metric_string)
    output += "Avg Nearest Metrics: {}\n".format(nearest_metric_string)

    average_result_mcf_psnr = result_mcf_psnr_all.mean(dim=0).detach().cpu().tolist()
    average_bilinear_mcf_psnr = bilinear_mcf_psnr_all.mean(dim=0).detach().cpu().tolist()
    average_nearest_mcf_psnr = nearest_mcf_psnr_all.mean(dim=0).detach().cpu().tolist()
    avg_thresholds = mcf_psnr.thresholds.tolist()

    save_ablation_linechart(results_dir, avg_thresholds, average_result_mcf_psnr, average_bilinear_mcf_psnr, average_nearest_mcf_psnr)

    output += "-------------------------------\n"
    output += "Result MCF PSNR: {}\n".format(average_result_mcf_psnr)
    output += "Bilinear MCF PSNR: {}\n".format(average_bilinear_mcf_psnr)
    output += "Nearest MCF PSNR: {}\n".format(average_nearest_mcf_psnr)
    output += "Thresholds: {}\n".format(avg_thresholds)

    with open(os.path.join(results_dir, "avg_result.txt"), "w") as f:
        f.write(output)

print(output)