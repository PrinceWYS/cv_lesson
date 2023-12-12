import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

def estimate(img1, img2, matches):
    if len(matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([ img1['kp'][int(m.queryIdx)].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ img2['kp'][int(m.trainIdx)].pt for m in matches ]).reshape(-1,1,2)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        print('[INFO] homography matrix is:\n', M)
        return M

    else:
        print("[ERROR] Not enough matches are found - {}/{}".format(len(matches), MIN_MATCH_COUNT) )
    return M