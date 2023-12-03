import numpy as np
import os
from PIL import Image

import imread_datadir_re as idr
  
def load_datadir_re(datadir, bitDepth, resize, gamma): 
    # Parse options
    load_imgs = True
    load_mask = True
    white_balance = np.array([1, 1, 1])
      
    if np.ndim(white_balance) == 1:
        white_balance = np.diag(white_balance)

    print('------------')
    print("white balance:\n", white_balance)

    # Build data struct  
    if isinstance(datadir, dict):
        print('------------')
        print("recreate data")
        data = datadir  
    elif isinstance(datadir, str):
        data = {}
        data['s'] = read_floats__(os.path.join(datadir, 'light_directions.txt'))
          
        data['L'] = read_floats__(os.path.join(datadir, 'light_intensities.txt'))
        data['L'] = np.dot(data['L'], white_balance)
          
        data['filenames'] = read_strings__(os.path.join(datadir, 'filenames.txt'))
        data['filenames'] = [os.path.join(datadir, x) for x in data['filenames']]
    else:
        raise ValueError('datadir is neither a struct nor a string!')
      
    # Load mask image
    if 'mask' not in data and load_mask:
        print('------------')
        print("load mask image")
        mask_path = os.path.join(datadir, 'mask.png')
        with Image.open(mask_path) as mask_img:
            # mask_img = mask_img.resize(resize, resample=Image.NEAREST)
            data['mask'] = np.array(mask_img)
        print('------------')
        # print("data[mask]\n", data['mask'])
        print('image\'s shape: ', data['mask'].shape)
      
    # Load images
    if 'imgs' not in data and load_imgs:
        print('------------')
        print("load images, total length is: ", len(data['filenames']))
        data_ = data.copy()
        data['imgs'] = []
        for i in range(len(data['filenames'])):
            img = idr.imread_datadir_re(data_, i, bitDepth, resize, gamma)
            data['imgs'].append(img)
        
    return data
  
def read_floats__(fn):
    with open(fn, 'rt') as fid:
        out = np.loadtxt(fid, dtype=float, comments=None)
    return out
  
def read_strings__(fn):
    with open(fn, 'rt') as fid:
        out = [line.rstrip() for line in fid]
    return out 