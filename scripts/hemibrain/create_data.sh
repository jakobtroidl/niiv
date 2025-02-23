python experiments/create_dataset.py \
    --name hemibrain-256-256-256-mip-1 \
    --path https://storage.googleapis.com/neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg \
    --image_size 256 \
    --n_vols 400 \
    --train \
    --mip 1 \