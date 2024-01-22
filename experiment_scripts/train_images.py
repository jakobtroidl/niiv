# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, utils, loss_functions, modules
# import training_gpu as training
import training

from torch.utils.data import DataLoader  # noqa: E402
import configargparse
from functools import partial
import shutil
import json

import wandb

command_line = ""
for arg in sys.argv:
    command_line += arg
    command_line += " "
    

p = configargparse.ArgumentParser()
p.add('-c', '--config', required=True,  help='Path to config file. (e.g., ./config/config_small.json)')
p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=False, default="",
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-2')
p.add_argument('--num_epochs', type=int, default=6000, help='Number of epochs to train for.')
p.add_argument('--epochs_til_ckpt', type=int, default=1000, help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=50,
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--dataset', type=str, required=True, help="Dataset Path, (e.g., /data/UVG/Jockey)")
p.add_argument('--batch_size', type=int, default=8, help="Batch size")
opt = p.parse_args()


with open(opt.config, 'r') as f:
    config = json.load(f)

video_path = opt.dataset

print("Data path is {}".format(video_path))

# Define the model.
model = modules.CVR(type='cvr', out_features=1, encoding_config=config["cvr"])
model.cuda()

image_dataset = dataio.ImageDataset(path_to_info=opt.dataset)
dataloader = DataLoader(image_dataset, shuffle=True, batch_size=opt.batch_size, num_workers=0)


params = utils.get_n_params(model) # TODO update this function

root_path = os.path.join(opt.logging_root, opt.experiment_name)
if os.path.exists(root_path):
    val = input("The model directory %s exists. Overwrite? (y/n)"%root_path)
    if val == 'y':
        shutil.rmtree(root_path)
    else:
        raise NotImplementedError("File exists Error: %s"%root_path)
os.makedirs(root_path, exist_ok=True)


config_save_path = os.path.join(opt.logging_root, opt.experiment_name, "config.json")
with open(config_save_path, "w") as json_file:
    json.dump(config, json_file, indent=4)

train_config = {
    "dataset": opt.dataset,
    "lr": opt.lr,
    "num_epochs": opt.num_epochs,
    "experiment_name": opt.experiment_name
}
train_config_save_path = os.path.join(opt.logging_root, opt.experiment_name, "train_config.json")
with open(train_config_save_path, "w") as json_file:
    json.dump(train_config , json_file, indent=4)

# Define the loss
loss_fn = partial(loss_functions.image_l1, None)
summary_fn = partial(utils.write_volume_summary, image_dataset)


results_dir = os.path.join(root_path, 'results')
os.makedirs(results_dir, exist_ok=True)

results_file_path = os.path.join(results_dir, "results.txt")
f = open(results_file_path, 'w')

f.write("#"*30+" Training Info "+"#"*30)
f.write(f"\n{command_line}")
f.write(f"\n - experiment name: {opt.experiment_name}") 
f.write(f"\n - video name: {opt.dataset}")
f.write(f"\n - epochs: {opt.num_epochs}")
f.write(f"\n - learning rate: {opt.lr}")
f.write("#"*30+" Training Info "+"#"*30)
f.close()

#open and read the file after the appending:
f = open(results_file_path, "r")
print(f.read())
f.close()

psnr = training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
                        steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                        model_dir=root_path, summary_fn=summary_fn, opt=opt)


f = open(results_file_path, 'a')
f.write(f"\n")
f.close()

#open and read the file after the appending:
f = open(results_file_path, "r")
print(f.read())
f.close()

