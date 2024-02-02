for i in {1..20}; do \
    python experiments/eval_image_sequences.py \
    --experiment_name hemibrain-volume-pos-enc-v2 \
    --dataset data/hemibrain-volume-denoised-large/info.json \
    --config ./config/config_cvr_s.json \
    --iteration $i; \
    sleep 3m; \
done