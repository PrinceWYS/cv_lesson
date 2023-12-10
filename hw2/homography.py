import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

def estimate(img1, img2, matches):
    if len(matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([ img1['kp'][int(m.queryIdx)].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ img2['kp'][int(m.trainIdx)].pt for m in matches ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        print('[INFO] homography matrix is:\n', M)
        # print(mask)
        # matchesMask = mask.ravel().tolist()
        h, w= img1['gray'].shape
        pts = np.float32([ [0,0], [0,h-1], [w-1,h-1], [w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        # img2['gray'] = cv2.polylines(img2['gray'], [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        # result = cv2.warpPerspective(img1['img'], M, (img1['img'].shape[1] + img2['img'].shape[1], img1['img'].shape[0]))
        # result[0:img2['img'].shape[0], 0:img2['img'].shape[1]] = img2['img']
        # cv2.namedWindow('homography', 0)
        # # cv2.resizeWindow('homography')
        # cv2.imshow('homography', result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imwrite('mid.JPG', result)
        return M

    else:
        print( "Not enough matches are found - {}/{}".format(len(matches), MIN_MATCH_COUNT) )
        matchesMask = None

    return M