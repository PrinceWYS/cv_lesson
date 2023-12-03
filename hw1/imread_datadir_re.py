import numpy as np
import cv2
from PIL import Image
  
def imread_datadir_re(datadir, which_image, bitDepth, resize, gamma):
    if 'imgs' in datadir:
        print("*********")
        E = datadir['imgs'][which_image]
    else:  
        E = cv2.imread(datadir['filenames'][which_image], -1) #BGR
        E = cv2.cvtColor(E, cv2.COLOR_BGR2RGB)
        E = (E / (2**bitDepth - 1)) ** gamma  # for float input, set bitDepth = 1
        
        H, W, C = E.shape
        # Normalize the image with light intensities
        intensity_normalized = 1 / datadir['L'][which_image]
        intensity_normalized = np.diag(intensity_normalized)
        E = np.reshape(E, (H * W, C)).dot(intensity_normalized)
        E = np.maximum(0, E.reshape(H, W, C))

    return E