for i in {1..30}; do \
    python experiments/eval_image_sequences.py \
    --experiment_name hemibrain-volume-20-octaves \
    --dataset data/hemibrain-volume-denoised-large/info.json \
    --config ./config/config_cvr_s.json \
    --iteration $i; \
    sleep 4m; \
done