
bsub -J "NIIV Hemibrain" -n 8 -gpu "num=1" -q gpu_l4 -o logs/hemibrain-zhang-small-32-dec-units.log python experiments/train_images.py \
    --experiment_name hemibrain-zhang-small-32-dec-units \
    --dataset /nrs/turaga/jakob/neural-volumes/data/hemibrain-volume-noisy-large/info.json \
    --config /groups/turaga/home/troidlj/neural-volumes/config/config_cvr_s.json \
    --num_epochs 2000 \
    --batch_size 50 \
    --lr 0.0005 \