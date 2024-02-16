# Assuming img is your image tensor of shape [B, C, H, W]
def crop_image_border(img, crop_border):
    """
    Crop the borders of an image.

    Parameters:
    - img: A tensor representing the image, shape [B, C, H, W].
    - crop_border: The number of pixels to crop from each border (top, bottom, left, right).

    Returns:
    - Cropped image tensor.
    """
    B, C, H, W = img.shape
    return img[:, :, crop_border:-crop_border, crop_border:-crop_border]

