
bsub -J "NIIV Hemibrain" -n 8 -gpu "num=1" -q gpu_h100 -o logs/hemibrain-ours-liconn.log python experiments/train_images.py \
    --experiment_name hemibrain-ours-liconn \
    --dataset /nrs/turaga/jakob/neural-volumes/data/liconn-400/info.json \
    --config /groups/turaga/home/troidlj/neural-volumes/config/config_cvr_tiny.json \
    --num_epochs 2000 \
    --batch_size 50 \
    --lr 0.0005 \