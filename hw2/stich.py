import numpy as np
import cv2
from copy import copy

def blend_images(img1, img2, w1, w2):
    assert img1.shape == img2.shape
    out = copy(img1)
    valid_mask1 = (img1 > 0).astype(bool)
    valid_mask2 = (img2 > 0).astype(bool)
    inter_mask = ((valid_mask2 + valid_mask1) > 1).astype(bool)
    kernel = cv2.getGaussianKernel(5, 1)
    sub = (valid_mask2.astype(int) - inter_mask.astype(int) > 0).astype(bool)
    # blending
    out[sub] = img2[sub]
    return out

def getWarpBound(w, h, M):
    pts = (np.float32([[0, 0],
                      [w-1, h-1]])).reshape((-1, 1, 2))
    pts_format = pts.shape[1:]
    pts = pts.reshape((-1, 2))
    pts3 = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    # M_tmp = M.reshape([3, 3])
    warp_dst_pts = (M @ pts3.T).T
    warp_dst_pts = warp_dst_pts[:, :2] / warp_dst_pts[:, 2].reshape((-1, 1))
    warp_pts = (warp_dst_pts.reshape((-1, *pts_format))).reshape((-1, 2))
    w_min = np.min([warp_pts[0, 0], 0])
    h_min = np.min([warp_pts[0, 1], 0])
    w_max = np.max([warp_pts[1, 0], w - 1])
    h_max = np.max([warp_pts[1, 1], h - 1])
    
    return w_min, h_min, w_max, h_max

def warp_src_image(img, M, low=None, high=None):
    if low is None:
        low = (0, 0)
    assert high is not None
    w_low, h_low = low
    w_high, h_high = high
    content_height = np.ceil(h_high - h_low).astype(int)
    content_width = np.ceil(w_high - w_low).astype(int)
    # if content_width * content_height > 3000 * 3000:
    #     raise Exception('exceed maximum image size')

    xx, yy = np.meshgrid(np.arange(w_low, w_high), np.arange(h_low, h_high))
    xx = xx.flatten().reshape((-1, 1))
    yy = yy.flatten().reshape((-1, 1))

    pos = np.concatenate([xx, yy, np.ones_like(yy)], axis=1)
    transformed_pts = (M @ pos.T).T
    transformed_pts = transformed_pts[:, :2] / transformed_pts[:, 2].reshape((-1, 1))

    transformed_pts = transformed_pts.reshape((content_height, content_width, 2))

    invalid_mask = ((transformed_pts[..., 0] < 0) + (transformed_pts[..., 1] < 0) \
                    + (transformed_pts[..., 0] > content_width - 1) + (
                            transformed_pts[..., 1] > content_height - 1)) > 0
    # map pixel (x, y) of each position
    warped_img = cv2.remap(img, transformed_pts[..., 0].astype(np.float32),
                           transformed_pts[..., 1].astype(np.float32),
                           interpolation=cv2.INTER_LINEAR)
    warped_img[invalid_mask] = 0
    return warped_img

def stitch_simple(img1, img2, M):
    horizontal = M[0][2]
    vertical = M[1][2]
    if abs(horizontal) > abs(vertical):
        print('[INFO] stich two pics from left and right')
        width = img1.shape[1] + img2.shape[1]
        height = max(img1.shape[0], img2.shape[0])
        if horizontal > 0:
            result = cv2.warpPerspective(img1, M, (img1.shape[1] + img2.shape[1], img1.shape[0]))
            result[0:img2.shape[0], 0:img2.shape[1]] = img2
        else:
            M_inv = np.linalg.inv(M)
            result = cv2.warpPerspective(img2, M_inv, (img1.shape[1] + img2.shape[1], img2.shape[0]))
            result[0:img1.shape[0], 0:img1.shape[1]] = img1
    else:
        print('[INFO] stich two pics from up and down')
        if vertical > 0:
            result = cv2.warpPerspective(img1, M, (img1.shape[1], img1.shape[0] + img2.shape[0]))
            result[0:img2.shape[0], 0:img2.shape[1]] = img2
        else:
            M_inv = np.linalg.inv(M)
            result = cv2.warpPerspective(img2, M_inv, (img1.shape[1], img1.shape[0] + img2.shape[0]))
            result[0:img1.shape[0], 0:img1.shape[1]] = img1
    return result

def stitch(img1, img2, M, low=None, high=None):
    h, w, _ = img1.shape
    w_min, h_min, w_max, h_max = getWarpBound(w, h, M)
    
    if high is None:
        high = (w_max, h_max)
    if low is None:
        low_offset_w, low_offset_h = (0, 0)
        if w_min < 0:
            low_offset_w = np.floor(w_min).astype(int)
        if h_min < 0:
            low_offset_h = np.floor(h_min).astype(int)
        low = (low_offset_w, low_offset_h)
    warp_img1 = warp_src_image(img1, np.linalg.inv(M), low=low, high=high)
    warp_img2 = warp_src_image(img2, np.eye(3), low=low, high=high)
    img = blend_images(warp_img1, warp_img2, 0.5, 0.5)
    
    return img