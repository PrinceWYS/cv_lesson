import numpy as np
import cv2
import sys

MIN_MATCH_COUNT = 10

def estimate(img1, img2, matches, iter):
    if len(matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([ img1['kp'][int(m.queryIdx)].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ img2['kp'][int(m.trainIdx)].pt for m in matches ]).reshape(-1,1,2)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        print(f'[INFO] homography matrix is:\n', M)
        # print(customized_find_homograph(src_pts, dst_pts))
        my_M = solve_homograph(src_pts, dst_pts, max_iteration=iter)
        print(f'[INFO] my homography matrix is:\n', my_M)
        return M, my_M

    else:
        print("[ERROR] Not enough matches are found - {}/{}".format(len(matches), MIN_MATCH_COUNT))
        sys.exit(1)

def solve_homograph(src_pts, dst_pts, epsilon=0.5, max_iteration=100):
    assert src_pts.shape[0] == dst_pts.shape[0]
    src_pts = np.squeeze(src_pts)
    dst_pts = np.squeeze(dst_pts)

    def calc_dist(x, y):
        return np.sqrt(np.sum((x - y) ** 2, axis=-1))

    best_inliers = set()
    estimated_M = None
    for i in range(max_iteration):
        # random select 4 pairs
        sampled_idx = np.random.permutation(src_pts.shape[0])[: 4]
        sampled_src = np.array(src_pts[sampled_idx])
        sampled_dst = np.array(dst_pts[sampled_idx])

        M = solveHomographyMatrix(sampled_src, sampled_dst)

        warp_src_pts = cv2.perspectiveTransform(np.float32(src_pts).reshape(-1, 1, 2), M).reshape(-1, 2)
        dist = calc_dist(warp_src_pts, dst_pts)

        inliers = {(pt[0], pt[1]) for i, pt in enumerate(src_pts) if dist[i] < epsilon}

        if src_pts.shape[0] == 4:
            return M
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            estimated_M = M

    if len(best_inliers) > src_pts.shape[0] / 2:
        return estimated_M
    else:
        sampled_idx = np.random.permutation(src_pts.shape[0])[: 4]
        sampled_src = np.array(src_pts[sampled_idx])
        sampled_dst = np.array(dst_pts[sampled_idx])
        solveHomographyMatrix(sampled_src, sampled_dst)

    return estimated_M

def solveHomographyMatrix(sampled_src, sampled_dst):
    A = []
    for (src, dst) in zip(sampled_src, sampled_dst):
        sx, sy = src[0], src[1]
        dst_x, dst_y = dst[0], dst[1]

        A.append([sx, sy, 1., 0, 0, 0, -dst_x * sx, -dst_x * sy, -dst_x])
        A.append([0, 0, 0, sx, sy, 1., -dst_y * sx, -dst_y * sy, -dst_y])

    A = np.array(A)
    _, _, V = np.linalg.svd(A)
    M = V[-1]  # the smallest eigenvector
    M = M.reshape((3, 3))
    M = (1. / M.item(8)) * M
    return M