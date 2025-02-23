bsub -J "NIIV Hemibrain" -n 8 -gpu "num=1" -q gpu_a100 -o logs/hemibrain-train-256-v4.log python experiments/train_images.py \
    --experiment_name hemibrain-train-256-v4 \
    --dataset /nrs/turaga/jakob/neural-volumes/data/hemibrain-256-256-256-mip-1/info.json \
    --config /groups/turaga/home/troidlj/neural-volumes/config/config_cvr_s.json \
    --num_epochs 5000 \
    --batch_size 7 \
    --lr 0.0005 \