import cv2
import numpy as np
from typing import Tuple


class ImageSequenceDisplay:
    def __init__(self, window_name: str='Image Sequence', window_size: Tuple[int, int]=(800, 800), rgb: bool=False):
        """
        Initialize ImageSequenceDisplay.

        Args:
            window_name : A string for the window name.
            window_size : A tuple for the window size. Format should be (width, height).
            rgb         : A boolean indicating if the input is in RGB format. If True, will convert to BGR for display.
        """
        self.window_name = window_name
        self.rgb         = rgb
        # ---
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, window_size[0], window_size[1])


    def display(self, images: np.ndarray, wait: int = 30, pause_step=None):
        """
        Display an image sequence. Assumes input is in the format (steps, width, height, channels).

        Args:
            images: An ndarray containing the images to display. Shape should be (steps, width, height, channels).
            wait: An integer that specifies the argument for cv2.waitKey function, which is delay between frames.
        """
        for i in range(images.shape[0]):
            img = images[i]
            if self.rgb:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.window_name, img)

            # Wait for specified ms. If 0, waits indefinitely for a key stroke
            if cv2.waitKey(wait) & 0xFF == ord('q'):
                break

            if i == pause_step:
                import ipdb; ipdb.set_trace()

    def close(self):
        """Destroy the display window."""
        cv2.destroyWindow(self.window_name)
