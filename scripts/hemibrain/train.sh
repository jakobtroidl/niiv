bsub -J "NIIV Hemibrain" -n 8 -gpu "num=1" -q gpu_h100 -o logs/hemibrain-train-feat-reg-v3.log python experiments/train_images.py \
    --experiment_name hemibrain-train-feat-reg-v3 \
    --dataset /nrs/turaga/jakob/neural-volumes/data/hemibrain-volume-noisy-large/info.json \
    --config /groups/turaga/home/troidlj/neural-volumes/config/config_cvr_s.json \
    --num_epochs 5000 \
    --batch_size 50 \
    --lr 0.0001 \