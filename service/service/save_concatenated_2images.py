import numpy as np
from PIL import Image
from torch_numpy_converter import to_numpy_uint8
import torch
import os



def convert(images):
    concatenated_image = torch.cat(images, dim=-1)
    concatenated_image = concatenated_image.squeeze(0)
    concatenated_image = to_numpy_uint8(concatenated_image * 255)
    concatenated_image = np.transpose(concatenated_image, (1, 2, 0))
    # concatenated_image = concatenated_image[..., ::-1]
    return concatenated_image



def save_concatenated_2images(images1, images2, filename, tag, target_position):
    """
    Concatenates a list of images (numpy arrays) horizontally and saves the result.

    Parameters:
    images (list of numpy arrays): List of image data arrays with shape (height, width, channels).
    filename (str): Filename to save the concatenated image.
    """
    # Check if the images list is empty
    if not images1:
        raise ValueError("The images list is empty.")

    # Check if all images have the same height and channel number
    # if is_torch:
    # heights  = [to_numpy_uint8(img.shape[0]*255) for img in images]
    # channels = [to_numpy_uint8(img.shape[2]*255) if len(img.shape) > 2 else 1 for img in images]
    # else:
    print("images1[0].shape = ", images1[0].shape)

    heights  = [img.shape[0] for img in images1]
    channels = [img.shape[2] if len(img.shape) > 2 else 1 for img in images1]

    if len(set(heights)) > 1 or len(set(channels)) > 1:
        raise ValueError("All images must have the same height and number of channels.")

    # --------- images1 : Concatenate the images ----------
    cat_image1         = convert(images1)
    cat_image2         = convert(images2)
    concatenated_image = np.concatenate((cat_image1, cat_image2), axis=0)

    # Save the concatenated image
    print("concatenated_image.shape = ", concatenated_image.shape)
    Image.fromarray(concatenated_image).save(filename)
    # ---
    name            = filename.split('/')[-1].split(".")[0]
    common_save_dir = "/".join(filename.split('/')[:-4]) + "/summary_icem/{}".format(name)
    print("common_save_dir =", common_save_dir)
    os.makedirs(common_save_dir, exist_ok=True)
    Image.fromarray(concatenated_image).save(
        os.path.join(common_save_dir,
            "{}[tag={}][target={}].pdf".format(name, tag, target_position))
    )
    return filename

if __name__ == '__main__':
# Example usage
# Assuming images is a list of numpy arrays of shape (height, width, channel)
    images = [torch.Tensor(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)) for _ in range(5)]
    save_concatenated_images(images, './concatenated_image.png')

