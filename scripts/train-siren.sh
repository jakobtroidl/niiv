
export PYTHONPATH="/path/to/your/module:$PYTHONPATH"
python benchmarks/siren/train.py \
    --experiment_name hemibrain-siren-v3 \
    --dataset data/hemibrain-volume-noisy-large/test/ \
    --config ./config/config_siren.json 