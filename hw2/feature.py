import numpy as np
import cv2
from tqdm import tqdm
import sys

def detection(img, detectionMethod, gui=False):
    if detectionMethod == "SIFT":
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(img['gray'], None)
        img['feature'] = np.zeros(img['img'].shape)
        img['feature'] = cv2.drawKeypoints(img['gray'], kp, img['feature'])
    elif detectionMethod == "ORB":
        orb = cv2.ORB_create(1500)
        kp, des = orb.detectAndCompute(img['gray'], None)
        img['feature'] = np.zeros(img['img'].shape)
        img['feature'] = cv2.drawKeypoints(img['gray'], kp, None, (0,0,255), 0)
    else :
        print('[ERROR] Unsupport method')
        sys.exit(1)
    
    if gui:
        cv2.namedWindow('feature', 0)
        cv2.resizeWindow('feature', 768, 1024)
        cv2.imshow('feature', np.vstack((img['feature'], img['img'])))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    img['kp'] = kp
    img['des'] = des

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y)**2))

def knn_match(des1, des2, k):
    matches = []
    for i, desc1 in tqdm(enumerate(des1), total=len(des1)):  
        desc1_tile = np.tile(desc1, (len(des2), 1))  
        distances = np.sqrt(np.sum((desc1_tile - des2)**2, axis=1))  
        sorted_indices = np.argsort(distances)  
        k_best_matches = [(index, distances[index]) for index in sorted_indices[:k]]  
        matches.append((i, k_best_matches))
    return matches

def matching(img1, img2, gui=False):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    # matches = bf.knnMatch(img1['des'], img2['des'], k=2)
    matches = knn_match(img1['des'], img2['des'], k=2)

    goodMatch = []
    # for m, n in matches:
    #     if m.distance < 0.50 * n.distance:
    #         goodMatch.append(m)
    for query_idx, knn_list in matches:  
        best_match, second_best_match = knn_list[0], knn_list[1]
        if best_match[1] < 0.7 * second_best_match[1]:
            goodMatch.append((query_idx, best_match[0]))

    matches_dmatch = [cv2.DMatch(index1, index2, 0) for index1, index2 in goodMatch]

    res = cv2.drawMatches(img1['gray'], img1['kp'], img2['gray'], img2['kp'], matches_dmatch, None, flags=2)
    
    if gui:
        cv2.namedWindow('match', 0)
        cv2.resizeWindow('match', 1536, 512)
        cv2.imshow('match', res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return matches_dmatch


def solve(img1, img2, useMethod='SIFT', gui=False):
    print('[INFO] The feature detection method is ', useMethod)
    detection(img1, useMethod, gui)
    detection(img2, useMethod, gui)
    matches = matching(img1, img2, gui)
    return matches