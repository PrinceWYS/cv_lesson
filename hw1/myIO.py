import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def storeNormal(normal, datadir, isSort):
    normal_map = normal
    normal_map = ((normal_map - np.min(normal_map)) / (np.max(normal_map) - np.min(normal_map)) * 255).astype(np.uint8)
    normal_map = cv2.cvtColor(normal_map, cv2.COLOR_RGB2BGR)
    if isSort:
        name = datadir+'/normal_sort_map.png'
    else:
        name = datadir+'/normal_map.png'
    cv2.imwrite(name, normal_map)

def storeAlbedo(albedo, datadir, isSort):
    albedo_map =albedo
    albedo_map = ((albedo_map - np.min(albedo_map)) / (np.max(albedo_map) - np.min(albedo_map)) * 255).astype(np.uint8)
    albedo_map = cv2.cvtColor(albedo_map, cv2.COLOR_RGB2BGR)
    if isSort:
        name = datadir+'/albedo_sort_map.png'
    else:
        name = datadir+'/albedo_map.png'
    cv2.imwrite(name, albedo_map)

def store(data, datadir, normal, albedo, normal_sort, albedo_sort, mask, bitdepth=16,  gamma=1, resize=1):
    print('------------')
    print("start save pictures")
    path = './output/' + datadir
    if not os.path.exists(path):
        os.makedirs(path)
    storeNormal(normal, path, False)
    storeAlbedo(albedo, path, False)
    
    storeNormal(normal_sort, path, True)
    storeAlbedo(albedo_sort, path, True)
    
    N = normal
    N_sort = normal_sort
    L = data['s']   # directions 96 * 3
    filenames = data['filenames']
    imgs = data['imgs'] # 96 * 512 * 612 * 3
    intensenty = data['L']
    
    num = len(filenames) # 96

    for i in range(num):
        pic = imgs[i]
        newPic = np.array(albedo * (N @ L[i].reshape(3, 1)), dtype=float) * intensenty[i]
        newPic_sort = np.array(albedo_sort * (N_sort @ L[i].reshape(3, 1)), dtype=float)  * intensenty[i]
        
        pic = pic * intensenty[i] * (2 ** bitdepth - 1) / gamma
        pic = ((pic - np.min(pic)) / (np.max(pic) - np.min(pic)) * 255).astype(np.uint8)
        pic=  cv2.cvtColor(pic, cv2.COLOR_RGB2BGR)
        
        newPic = newPic * (2 ** bitdepth - 1) / gamma
        newPic = ((newPic - np.min(newPic)) / (np.max(newPic) - np.min(newPic)) * 255).astype(np.uint8)
        newPic = cv2.cvtColor(newPic, cv2.COLOR_RGB2BGR)
        
        newPic_sort = newPic_sort * (2 ** bitdepth - 1) / gamma
        newPic_sort = ((newPic_sort - np.min(newPic_sort)) / (np.max(newPic_sort) - np.min(newPic_sort)) * 255).astype(np.uint8)
        newPic_sort = cv2.cvtColor(newPic_sort, cv2.COLOR_RGB2BGR)
        
        newPic = newPic * np.expand_dims(mask, -1)
        newPic_sort = newPic_sort * np.expand_dims(mask, -1)
        
        # print(path+'/'+filenames[i][-7:])
        file = path+'/'+filenames[i][-7:]
        cv2.imwrite(file, np.hstack((pic, newPic, newPic_sort)))
