import read
import solveKP
import cv2
import sys

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if __name__ == "__main__":
    read.readVideo("./data/")
    kps = solveKP.getKeypoints("./data/output/0.jpg")
    