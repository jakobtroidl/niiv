python experiments/create_dataset.py \
    --name hemibrain-volume-denoised-large-v2 \
    --path https://storage.googleapis.com/neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg \
    --image_size 128 \
    --n_images 400 \
    --z \
    --anistropic_dim 2 \
    --denoise