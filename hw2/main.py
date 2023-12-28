import feature
import homography
import stitch
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import os
import time
from copy import copy

class MyStich:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("-img1", "--img1", help="image 1", dest="Img1", type=str, default="0")
        parser.add_argument("-img2", "--img2", help="image 2", dest="Img2", type=str, default="1")
        parser.add_argument("-d", "--dir", help="dir of pics need to stich", dest="Dir", type=str, default="./data1/")
        parser.add_argument("-m", "--method", help="feature detector method", dest="Method", type=str, default="SIFT")
        parser.add_argument("-a", "--all", help="whether stitch all pics", dest="ifAll", type=str, default="False")
        parser.add_argument("-g", "--gui", help="use gui mode", dest="Gui", type=bool, default=False)
        parser.add_argument("-i", "--iter", help="max iteration", dest="Iter", type=int, default=100)
        self.args = parser.parse_args()
        
        self.imgs = []
        self.imgName1 = self.args.Img1
        self.imgName2 = self.args.Img2
        self.dir = self.args.Dir
        self.method = self.args.Method.upper()
        self.ifAll = self.args.ifAll
        self.gui = self.args.Gui
        self.iter = self.args.Iter
        
        if self.dir.endswith('/'):  
            self.dir = self.dir[:-1]
        
        if self.ifAll == 'True':
            self.readImgAll()
        elif self.ifAll == 'False':
            self.img1 = self.readImg(self.imgName1)
            self.img2 = self.readImg(self.imgName2)
    
    def readImg(self, input):
        print(f'[INFO] Read image: ', input)
        data = {}
        data['img'] = cv2.imread(input)
        data['gray'] = cv2.cvtColor(data['img'], cv2.COLOR_BGR2GRAY)
        return data
    
    def readImgAll(self):
        filelist = os.listdir(self.dir)
        for item in filelist:
            path = self.dir + '/' + item
            print(f'[INFO] Read image: ', path)
            data = {}
            data['img'] = cv2.imread(path)
            data['gray'] = cv2.cvtColor(data['img'], cv2.COLOR_BGR2GRAY)
            self.imgs.append(data)

if __name__ == "__main__":
    start = time.time()
    myStich = MyStich()

    if myStich.ifAll == 'True':
        n_image = len(myStich.imgs)
        M_list = [0] * n_image
        M_list[0] = np.eye(3)
        
        for i in range(1, n_image):
            cur_img = myStich.imgs[i]
            pre_img = myStich.imgs[i-1]
            
            # stitch to prev
            matches = feature.solve(cur_img, pre_img, myStich.method, myStich.gui)
            _, M = homography.estimate(cur_img, pre_img, matches, myStich.iter)
            kp1 = cur_img['kp']
            kp2 = pre_img['kp']
            
            M_list[i] = M_list[i-1] @ M
        
        # pick up a pivot
        pivotIdx = n_image // 2
        warp_bounds = [0] * n_image
        M_pivot = copy(M_list[pivotIdx])
        
        for i in range(n_image):
            M_list[i] = np.linalg.inv(M_pivot) @ M_list[i]
            h, w, _ = myStich.imgs[i]['img'].shape
            warp_bounds[i] = stitch.getWarpBound(w, h, M_list[i])
        global_w_min, global_h_min, global_w_max, global_h_max = warp_bounds[0]
        
        for i in range(1, n_image):
            w_min, h_min, w_max, h_max = warp_bounds[i]
            global_w_min = min(global_w_min, w_min)
            global_h_min = min(global_h_min, h_min)
            global_w_max = max(global_w_max, w_max)
            global_h_max = min(global_h_max, h_max)
            
        
        result = np.zeros((
            np.ceil(global_h_max - global_h_min).astype(int),
            np.ceil(global_w_max - global_w_min).astype(int),
            3), dtype=myStich.imgs[0]['img'].dtype)
        
        print(f'[INFO] Start to stitch...')
        
        for i in tqdm(range(n_image)):
            img1 = myStich.imgs[i]['img']
            img2 = myStich.imgs[pivotIdx]['img']
            warp_img = stitch.stitch(img1, img2, M_list[i], (global_w_min, global_h_min), (global_w_max, global_h_max))
            result = stitch.blend_images(result, warp_img)
    else:
        matches = feature.solve(myStich.img1, myStich.img2, myStich.method, myStich.gui)
        M, my_M = homography.estimate(myStich.img1, myStich.img2, matches, myStich.iter)
        result = stitch.stitch(myStich.img1['img'], myStich.img2['img'], my_M)

    print(f'[INFO] result size: ', result.shape)
    end = time.time()
    print(f'[INFO] ', myStich.method, ' takes ', end-start, ' seconds')
    cv2.namedWindow('Result', 0)
    cv2.imshow('Result', result)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()
    cv2.imwrite('output.JPG', result)