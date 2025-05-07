
bsub -J "NIIV Hemibrain" -n 8 -gpu "num=1" -q gpu_a100 -o logs/hemibrain-zhang-fafb.log python experiments/train_images.py \
    --experiment_name hemibrain-zhang-fafb \
    --dataset /nrs/turaga/jakob/neural-volumes/data/fafb-130/info.json \
    --config /groups/turaga/home/troidlj/neural-volumes/config/config_cvr_s.json \
    --num_epochs 2000 \
    --batch_size 50 \
    --lr 0.0005 \