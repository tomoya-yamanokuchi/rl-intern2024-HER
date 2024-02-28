from PIL import Image
from .reverse_channel import reverse_channel


def create_gif_from_Image_list(
        images  : list,
        fname   : str  = './output.gif',
        duration: int  = 500,
        rgb     : bool = True,
    ):
    '''
    - transparency:
            Transparency color index.
    - duration:
            The display duration of each frame of the multiframe gif, in milliseconds.
            Pass a single integer for a constant duration, or a list or tuple to set the duration for each frame separately.
    - loop:
            Integer number of times the GIF should loop.
            0 means that it will loop forever.
            By default, the image will not loop.
    '''

    PIL_images = []
    for img in images:
        if not rgb:
        #    print(" +++  reverse_channel +++ \n ")
           img = reverse_channel(img)
        PIL_images.append(Image.fromarray(img))

    PIL_images[0].save(
        fname,
        save_all      = True,
        append_images = PIL_images[1:],
        optimize      = False,
        duration      = duration,
        loop          = 0
    )
