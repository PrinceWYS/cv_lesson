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
        orb = cv2.ORB_create(5000)
        kp, des = orb.detectAndCompute(img['gray'], None)
        img['feature'] = np.zeros(img['img'].shape)
        img['feature'] = cv2.drawKeypoints(img['gray'], kp, None, (0,0,255), 0)
    elif detectionMethod == "MYSIFT":
        mySift = cv2.SIFT_create()
        kp, des = mySift.detectAndCompute(img['gray'], None)
        num = len(kp)
        img['feature'] = np.zeros(img['img'].shape)
        img['feature'] = cv2.drawKeypoints(img['gray'], kp, img['feature'])
        des = np.zeros((num, 9))
        h, w = img['gray'].shape
        for i in range(len(kp)):
            y, x = kp[i].pt
            x = int(x)
            y = int(y)
            if x==0 and y==0:
                vec = np.array([0, 0,                   0,
                                0, img['gray'][x][y  ], img['gray'][x+1][y  ],
                                0, img['gray'][x][y+1], img['gray'][x+1][y+1]])
            elif x==h-1 and y==0:
                vec = np.array([0,                     0,                   0,
                                img['gray'][x-1][y  ], img['gray'][x][y  ], 0,
                                img['gray'][x-1][y+1], img['gray'][x][y+1], 0])
            elif x==0 and y==w-1:
                vec = np.array([0, img['gray'][x][y-1], img['gray'][x+1][y-1],
                                0, img['gray'][x][y  ], img['gray'][x+1][y  ],
                                0, 0,                   0])
            elif x==h-1 and y==w-1:
                vec = np.array([img['gray'][x-1][y-1], img['gray'][x][y-1], 0,
                                img['gray'][x-1][y  ], img['gray'][x][y  ], 0,
                                0, 0, 0])
            elif x==0:
                vec = np.array([0, img['gray'][x][y-1], img['gray'][x+1][y-1],
                                0, img['gray'][x][y  ], img['gray'][x+1][y  ],
                                0, img['gray'][x][y+1], img['gray'][x+1][y+1]])
            elif x==h-1:
                vec = np.array([img['gray'][x-1][y-1], img['gray'][x][y-1], 0,
                                img['gray'][x-1][y  ], img['gray'][x][y  ], 0,
                                img['gray'][x-1][y+1], img['gray'][x][y+1], 0])
            elif y==0:
                vec = np.array([0,                     0,                   0,
                                img['gray'][x-1][y  ], img['gray'][x][y  ], img['gray'][x+1][y  ],
                                img['gray'][x-1][y+1], img['gray'][x][y+1], img['gray'][x+1][y+1]])
            elif y==w-1:
                vec = np.array([img['gray'][x-1][y-1], img['gray'][x][y-1], img['gray'][x+1][y-1],
                                img['gray'][x-1][y  ], img['gray'][x][y  ], img['gray'][x+1][y  ],
                                0,                     0,                   0])
            else:
                vec = np.array([img['gray'][x-1][y-1], img['gray'][x][y-1], img['gray'][x+1][y-1],
                                img['gray'][x-1][y  ], img['gray'][x][y  ], img['gray'][x+1][y  ],
                                img['gray'][x-1][y+1], img['gray'][x][y+1], img['gray'][x+1][y+1]])
            des[i] = vec
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

def distance(x, y, type="SIFT"):
    if type == "SIFT" or type == "MYSIFT":
        return np.sqrt(np.sum((x - y)**2, axis=1))
    elif type == "ORB":
        return np.sum(x != y, axis=1)

def knn_match(des1, des2, k, type="SIFT"):
    matches = []
    for i, desc1 in tqdm(enumerate(des1), total=len(des1)):
        desc1_tile = np.tile(desc1, (len(des2), 1))  
        # distances = np.sqrt(np.sum((desc1_tile - des2)**2, axis=1))
        distances = distance(desc1_tile, des2, type)
        sorted_indices = np.argsort(distances)
        k_best_matches = [(index, distances[index]) for index in sorted_indices[:k]]
        matches.append((i, k_best_matches))
    return matches

def matching(img1, img2, gui=False, type="SIFT"):
    matches = knn_match(img1['des'], img2['des'], k=2, type=type)

    goodMatch = []
    for query_idx, knn_list in matches:  
        best_match, second_best_match = knn_list[0], knn_list[1]
        if best_match[1] < 0.7 * second_best_match[1]:
            goodMatch.append((query_idx, best_match[0]))

    print(f'[INFO] Num of good match: ', len(goodMatch))
    matches_dmatch = [cv2.DMatch(index1, index2, 0) for index1, index2 in goodMatch]

    res = cv2.drawMatches(img1['gray'], img1['kp'], img2['gray'], img2['kp'], matches_dmatch, None, flags=2)
    
    cv2.imwrite('match.jpg', res)
    
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
    matches = matching(img1, img2, gui, useMethod)
    return matches