import feature
import homography
import stich
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
    parser.add_argument("-m", "--method", help="feature detector method", dest="Method", type=str, default="SIFT")
    parser.add_argument("-a", "--all", help="whether stich all pics", dest="ifAll", type=bool, default="False")
    args = parser.parse_args()
    
    imgName1 = args.Img1
    imgName2 = args.Img2
    method = args.Method.upper()
    ifAll = args.ifAll

    if ifAll:
        pass
    else:
        readImg(img1, imgName1)
        readImg(img2, imgName2)
        
        matches = feature.solve(img1, img2, method)
        M = homography.estimate(img1, img2, matches)
        # result = stich.stich(img1['img'], img2['img'], M)
        result = stich.stitch_image(img1['img'], img2['img'], M)
    print('[INFO] result size: ', result.shape)
    cv2.namedWindow('Result', 0)
    cv2.imshow('Result', result)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()
    cv2.imwrite('output.JPG', result)