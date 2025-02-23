import argparse
from cloudvolume import CloudVolume
import numpy as np
import os
from tqdm import tqdm
from skimage.restoration import denoise_tv_chambolle
import json
import torch.nn.functional as F
import torch

def main(args):

    # load cloudvolume
    cv = CloudVolume(
        args.path,
        cache=True,
        parallel=True,
    )
    mip = args.mip

    try:
        volume_size = cv.info["scales"][mip]["size"]
        resolution = cv.info["scales"][mip]["resolution"]
        print("Volume size: ", volume_size)
        print("Resolution: ", resolution)
    except KeyError:
        print("Double check if the neuroglancer precomputed info file contains necessary metadata.")

    output_path = "./data/{}/".format(args.name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if args.train:
        output_path = os.path.join(output_path, "train")
    else:
        output_path = os.path.join(output_path, "test")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    n_vols = args.n_vols

    if args.coord_list_path:
        with open(args.coord_list_path, 'r') as file:
            data = json.load(file)
        n_vols = len(data)

    # load dataset
    for i in tqdm(range(n_vols)):
        download_size = [args.image_size, args.image_size, args.image_size]

        if args.coord_list_path:
            x, y, z = data[i]
        else:
            x = np.random.randint(15_000, 21_000) # assuming x,y is the high resolution axis
            y = np.random.randint(15_000, 21_000)
            z = np.random.randint(15_000, 21_000)
            # x = 3859
            # # x = np.random.randint(10_000, 100_000)
            # y = np.random.randint(3_000, 7_000)
            # z = np.random.randint(400, 1_200)
            
        point = [x, y, z]

        # get image
        vol = cv.download_point(tuple(point), size=tuple(download_size), mip=mip)

        if args.denoise:
            vol_denoised = denoise_tv_chambolle(vol, weight=0.1, channel_axis=-1)
            vol_denoised = np.squeeze(vol_denoised * 255.0).astype(np.uint8)
             # save numpy array to disk
            np.save(os.path.join(output_path, "{}_{}_{}_denoised.npy".format(x, y, z)), vol_denoised)
        else:
            # save numpy array to disk   
            vol = np.squeeze(vol)

            # interpolate F to shape of [128, 128, 128]
            vol = vol.astype(np.float32)
            vol = vol / 255.0
            vol = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)
            # vol = F.interpolate(vol, size=(args.image_size, args.image_size, args.image_size), mode='trilinear')

            vol = vol.squeeze().detach().cpu().numpy()
            vol = vol * 255.0
            vol = vol.astype(np.uint8)
            
            np.save(os.path.join(output_path, "{}_{}_{}.npy".format(x, y, z)), vol)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example Script")
    parser.add_argument('--name', type=str, help='Name of dataset', required=True)
    parser.add_argument('--path', type=str, help='Path to ng precomputed file', required=True)
    parser.add_argument('--train', action="store_true", help='Whether to download training or test images')
    parser.add_argument('--image_size', type=int, help='Pixel size of dataset images', default=128, required=True)
    parser.add_argument('--denoise', action="store_true", help='Whether to apply variation denoising (TV)')
    parser.add_argument('--coord_list_path', type=str, help='List of 3D coordinates that contain volume centers', default=None, required=False)
    parser.add_argument('--n_vols', type=int, help='Number of images being downloaded', default=200, required=True)
    parser.add_argument('--mip', type=int, help='MIP level of the dataset', default=0, required=False)


    args = parser.parse_args()
    main(args)