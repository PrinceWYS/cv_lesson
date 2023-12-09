import feature
import homography
import numpy as np
import cv2
import sys
import argparse

def readImg(data, input):
    print('[INFO] Read image: ', input)
    data['img'] = cv2.imread(input)
    data['gray'] = cv2.cvtColor(data['img'], cv2.COLOR_BGR2GRAY)

if __name__ == "__main__":
    img1 = {}
    img2 = {}

    parser = argparse.ArgumentParser()
    parser.add_argument("-img1", "--img1", help="image 1", dest="Img1", type=str, default="0")
    parser.add_argument("-img2", "--img2", help="image 2", dest="Img2", type=str, default="1")
    parser.add_argument("-m", "--method", help="this is method", dest="Method", type=str, default="SIFT")
    args = parser.parse_args()
    
    imgName1 = args.Img1
    imgName2 = args.Img2
    method = args.Method.upper()
    
    readImg(img1, imgName1)
    readImg(img2, imgName2)
    
    matches = feature.solve(img1, img2, method)
    homography.estimate(img1, img2, matches)