
bsub -J "NIIV Hemibrain" -n 8 -gpu "num=1" -q gpu_a100 -o logs/hemibrain-attn.log python experiments/train_images.py \
    --experiment_name hemibrain-attn \
    --dataset /nrs/turaga/jakob/neural-volumes/data/hemibrain-volume-noisy-large/info.json \
    --config /groups/turaga/home/troidlj/neural-volumes/config/config_cvr_s.json \
    --num_epochs 2000 \
    --batch_size 10 \
    --lr 0.0005 \