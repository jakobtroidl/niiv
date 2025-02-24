
bsub -J "NIIV Hemibrain" -n 8 -gpu "num=1" -q gpu_a100 -o logs/hemibrain-unfold-attempt-25-neighb.log python experiments/train_images.py \
    --experiment_name hemibrain-unfold-attempt-25-neighb \
    --dataset /nrs/turaga/jakob/neural-volumes/data/hemibrain-volume-noisy-large/info.json \
    --config /groups/turaga/home/troidlj/neural-volumes/config/config_cvr_tiny.json \
    --num_epochs 2000 \
    --batch_size 50 \
    --lr 0.0005 \