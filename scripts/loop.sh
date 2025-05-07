for i in {1..30}; do \
    python niiv/eval_image_sequences.py \
    --experiment_name hemibrain-volume-yiqing-test \
    --dataset data/hemibrain-volume-denoised-large/info.json \
    --config ./config/config_cvr_s.json \
    --iteration $i; \
    sleep 2m; \
done