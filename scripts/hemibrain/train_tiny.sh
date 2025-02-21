bsub -J "NIIV Hemibrain" -n 8 -gpu "num=1" -q gpu_h100 -o logs/hemibrain-train-model-tiny-v3-no-attn.log python experiments/train_images.py \
    --experiment_name hemibrain-train-model-tiny-v3-no-attn \
    --dataset /nrs/turaga/jakob/neural-volumes/data/hemibrain-volume-noisy-large/info.json \
    --config /groups/turaga/home/troidlj/neural-volumes/config/config_cvr_tiny.json \
    --num_epochs 5000 \
    --batch_size 50 \
    --lr 0.0005 \