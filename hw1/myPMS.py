import numpy as np
import cv2

def L2_PMS(data, m):
    print('------------')
    print("start calculate normal")
    
    height, width, color = data['imgs'][0].shape
    normal = np.zeros((height, width, color))
    albedo = np.zeros((height, width, color))
    
    imgs = np.array(data['imgs'])
    L = data['s']

    for i in range(len(m[0])):
        h = m[0][i]
        w = m[1][i]

        N = (np.linalg.pinv(L) @ imgs[:, h, w]).T

        rho = np.linalg.norm(N, axis=1)
        albedo[h][w] = rho
        N_gray = N[0] * 0.299 + N[1] * 0.587 + N[2] * 0.114
        Nnorm = np.linalg.norm(N, axis=1)
        normal[h][w] = N_gray/(Nnorm + 1e-4)

    # albedo = (albedo - np.min(albedo))/(np.max(albedo)-np.min(albedo))
    
    return normal, albedo

def sort_data_PMS(data, m):
    print('------------')
    print("start calculate normal")
    
    height, width, color = data['imgs'][0].shape
    normal = np.zeros((height, width, color))
    albedo = np.zeros((height, width, color))
    
    imgs = np.array(data['imgs'])
    L = data['s']
    gray = imgs[:, :, :, 0] * 0.299 + imgs[:, :, :, 1] * 0.587 + imgs[:, :, :, 2]
    print(gray.shape)
    
    for i in range(len(m[0])):
        h = m[0][i]
        w = m[1][i]
        
        maxGray = np.max(gray[:, h, w])
        minGray = np.min(gray[:, h, w])
        
        L_sorted = np.array([])
        img_sorted = np.array([])
        
        for j in range(len(imgs)):
            if gray[j, h, w]<=minGray or gray[j, h, w]>=maxGray:
                continue
            else:
                L_sorted = np.append(L_sorted, L[j])
                img_sorted = np.append(img_sorted, imgs[j][h][w])
        
        num = len(L_sorted) // 3
        L_sorted = np.reshape(L_sorted, (num, 3))
        img_sorted = np.reshape(img_sorted, (num, 3))
        
        N = (np.linalg.pinv(L_sorted) @ img_sorted).T

        rho = np.linalg.norm(N, axis=1)
        albedo[h][w] = rho
        N_gray = N[0]*0.299 + N[1]*0.587 + N[2]*0.114
        Nnorm = np.linalg.norm(N, axis=1)
        normal[h][w] = N_gray/(Nnorm + 1e-4)

    # albedo = (albedo - np.min(albedo))/(np.max(albedo)-np.min(albedo))
    
    return normal, albedo