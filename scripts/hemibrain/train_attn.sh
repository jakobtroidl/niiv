bsub -J "AE Train (Topology)" -n 8 -gpu "num=1" -q gpu_h100 -o logs/hemibrain_train_attn_v0.log python experiments/train_niiv_attn.py \
    --experiment_name hemibrain-train-attn \
    --dataset /nrs/turaga/jakob/neural-volumes/data/hemibrain-volume-noisy-large/info.json \
    --config /groups/turaga/home/troidlj/neural-volumes//config/config_cvr_s_attn.json \
    --num_epochs 5000 \
    --batch_size 250 \
    --lr 0.0001 \