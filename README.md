# Neural Implicit Isotropic Volume Reconstruction

## Getting Started

```
conda create -n niv python=3.9
conda activate niv
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install igneous-pipeline pytorch-ignite tqdm wandb
```


## Start and Reconnect to Training Job using tmux

```bash
tmux new -s my_training_session
python train.py -opt options/train/train_iso_em.yml
```

Detach from the session using `Ctrl+B D`. 

Reconnect to the session using
```bash
tmux attach -t my_training_session
```

List existing tmux sessions
```bash
tmux list-sessions
```

Delete tmux session
```bash
tmux kill-session -t session-name
```

## Data Access

Requires `gsutil` command-line utility installed. See instructions [here](https://cloud.google.com/storage/docs/gsutil_install#linux).

Download public training data from GCS
```bash
cd neural-volumes
gsutil cp gs://neural-implicit-volumes/datasets/hemibrain-volume-denoised-large.zip ./data
cd data
unzip hemibrain-volume-denoised-large.zip
```

Convert reconstructed volume to NG precomputed and upload to GCS
```bash
igneous image create ./DATA.npy ./PRECOMPUTED_FOLDER --compress none
gsutil -m cp -r ./PRECOMPUTED_FOLDER/  gs://neural-implicit-volumes/NAME/
```











## References
We used the code from following repositories: [NVP](https://github.com/subin-kim-cv/NVP), [LIIF](https://github.com/yinboc/liif), [VINR](https://github.com/Picsart-AI-Research/VideoINR-Continuous-Space-Time-Super-Resolution).
