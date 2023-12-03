import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy

import load_datadir_re as ldr
import myPMS
import myIO

if __name__ == '__main__':
    dataFormat = 'PNG'
    dataNameStack = ['bear', 'cat', 'pot', 'buddha']

    for testId in range(4):
        print("start", testId)
        dataName = dataNameStack[testId] + dataFormat
        datadir = './pmsData/' + dataName
        bitdepth = 16
        gamma = 1
        resize = 1
        data = ldr.load_datadir_re(datadir, bitdepth, resize, gamma)
    
        L = data['s']
        f = L.shape[0]
        if np.ndim(data['mask']) == 2:
            color = 1
            height, width = data['mask'].shape
        else:
            height, width, color = data['mask'].shape
        
        if color == 1:
            mask1 = data['mask'] / 255.0
        else:
            mask1 = cv2.cvtColor(data['mask'], cv2.COLOR_BGR2GRAY) / 255.0
        mask3 = np.tile(mask1, (1, 1, 3))
        m = np.nonzero(mask1 == 1)
        p = len(m[0])
        print("shape of m: ", m[0].shape, ", p: ", p)
    
        # Standard photometric stereo
        normal, albedo = myPMS.L2_PMS(data, m)
        normal_sort, albedo_sort = myPMS.sort_data_PMS(data, m)
        
        myIO.store(data, dataName, normal, albedo, normal_sort, albedo_sort)