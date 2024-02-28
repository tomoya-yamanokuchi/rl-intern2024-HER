import torch
from torch import Tensor
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import Tensor
import matplotlib.pyplot as plt

import torch
from torch import Tensor
import matplotlib.pyplot as plt



def generate_test_tensor(batch_size: int, steps: int, width: int, height: int) -> Tensor:
    tensor = torch.zeros(batch_size, steps, 3, width, height)

    for b in range(batch_size):
        color = plt.cm.jet(b / (batch_size - 1))[:3] # Use a colormap to get a unique color for each batch
        for s in range(steps):
            alpha = s / (steps - 1) # Vary transparency across steps
            color_tensor = torch.tensor([c * alpha for c in color])
            tensor[b, s, :, :, :] = color_tensor.view(3, 1, 1).expand(3, width, height)

    return tensor


def flatten_image_sequence(tensor: Tensor):
    batch_size, step, channel, width, height = tensor.shape
    padding = 1  # Set the padding size

    # Create a padded tensor to insert black pixels
    padded_tensor = torch.zeros(batch_size, step, channel, width + 2 * padding, height + 2 * padding)
    padded_tensor[:, :, :, padding:width + padding, padding:height + padding] = tensor

    # Reshape tensor to create a large image for each step
    reshaped_tensor = padded_tensor.permute(1, 0, 2, 3, 4).contiguous().view(-1, channel, width + 2 * padding, height + 2 * padding)

    # Concatenate images along the width dimension for each step
    step_concat = [torch.cat([reshaped_tensor[s * batch_size + b] for b in range(batch_size)], dim=1) for s in range(step)]
    step_concat_tensor = torch.stack(step_concat, dim=0) # shape: (step, channel, (width + 2 * padding), batch_size * (height + 2 * padding))

    # Concatenate all step images along the height dimension
    final_image = torch.cat(tuple(step_concat_tensor), dim=2) # shape: (channel, (width + 2 * padding) * step, batch_size * (height + 2 * padding))

    return final_image




if __name__ == '__main__':
    batch_size = 10
    steps = 5
    width = 64
    height = 64

    # Generate a test tensor
    input_tensor = generate_test_tensor(batch_size, steps, width, height)

    # Process the tensor
    processed_image = flatten_image_sequence(input_tensor)

    # Save image using torchvision's save_image
    save_image(processed_image, "./file_path.png")
