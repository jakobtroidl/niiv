import numpy as np
import cv2
import tqdm

a = np.load(
    "/home/jakobtroidl/Desktop/neural-volumes/data/ista/20230716_ExPID114_17.50.53.npy"
)

out_path = '/home/jakobtroidl/Desktop/neural-volumes/data/ista/20230716_ExPID114_17.50.53_clahe.npy'

print(a.shape)
print(a.dtype)
print(a.min(), a.max())

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(127,127))
transformed_volume = np.zeros_like(a)

# Iterate over each slice along the first dimension (Z-axis)
for i in tqdm(range(a.shape[0])):
    # Assuming the transform is CLAHE for demonstration; replace with your transform.
    slice = clahe.apply(a[i, :, :])
    transformed_volume[i, :, :] = slice

np.save(out_path, transformed_volume)