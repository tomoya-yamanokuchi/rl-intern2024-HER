import numpy as np
from .create_gif_from_Image_list import create_gif_from_Image_list
import os

def create_gif_from_Image_array(
        images_array : np.ndarray,
        fname        : str = './output.gif',
        duration     : int = 500,
        tag : str = None,
        target_position = None,
    ):
    step, width, height, channel = images_array.shape
    images_list                  = np.array_split(images_array, indices_or_sections=step, axis=0)
    for i in range(len(images_list)):
        '''
         - np.array_split() では splitのaxisの次元が残ったままなので (1, width, height, channel)
           というように4次元のままで, create_gif_from_Image_listに渡すことができない
         - なので全ての要素の0次元目をsqueezeして置き換える処理を行う
        '''
        images_list[i] = images_list[i].squeeze(0)
    # print("images_list[0].shape = ", images_list[0].shape)
    create_gif_from_Image_list(images_list, fname, duration, rgb=True)


    # ----
    if tag is not None:
      name            = fname.split('/')[-1].split(".")[0]
      common_save_dir = "/".join(fname.split('/')[:-4]) + "/summary_icem/{}".format(name)
      print("common_save_dir =", common_save_dir)
      os.makedirs(common_save_dir, exist_ok=True)
      create_gif_from_Image_list(
          images      = images_list,
          duration    = duration,
          rgb         = True,
          fname       = os.path.join(common_save_dir,
              "{}[tag={}][target={}].gif".format(name, tag, target_position)
          )
      )
