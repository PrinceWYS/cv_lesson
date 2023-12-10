import numpy as np
import cv2

def stich(img1, img2, M):
    horizontal = M[0][2]
    vertical = M[1][2]
    if abs(horizontal) > abs(vertical):
        print('[INFO] stich two pics from left and right')
        width = img1.shape[1] + img2.shape[1]
        height = max(img1.shape[0], img2.shape[0])
        if horizontal > 0:
            result = cv2.warpPerspective(img1, M, (img1.shape[1] + img2.shape[1], img1.shape[0]))
            # result[0:img2.shape[0], 0:img2.shape[1]] = img2
            result = cv2.addWeighted(result, 0.5, img2, 0.5, 0)
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


def stitch_image(img1, img2, H):
    # 1. 获得每张图片的四个角点
    # 2. 对图片进行变换（单应性矩阵使图进行旋转，平移）
    # 3. 创建一张大图，将两张图拼接到一起
    # 4. 将结果输出

    #获得原始图的高/宽
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # 获取图片的四个角点
    img1_dims = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    img2_dims = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    # 将图1的原始四个点，根据单应性矩阵，获得投影坐标
    img1_transform = cv2.perspectiveTransform(img1_dims, H)

    # 将两个图像的角点拼接起来。
    result_dims = np.concatenate((img2_dims, img1_transform), axis=0)
    #print(result_dims)

    # 获取图像中的最小点，最大点，防止有些信息显示不到
    [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel()-0.5)
    [x_max, y_max ] = np.int32(result_dims.max(axis=0).ravel()+0.5)

    #平移的距离（左加右减，上加下减）
    transform_dist = [-x_min, -y_min]

    #[1, 0, dx]
    #[0, 1, dy]
    #[0, 0, 1 ]
    # 创建好平移矩阵
    transform_array = np.array([[1, 0, transform_dist[0]],
                                [0, 1, transform_dist[1]],
                                [0, 0, 1]])
    # 透视变换，得到结果矩阵（只是img1来进行变换），但是图片是img1+img2的大小
    result_img = cv2.warpPerspective(img1, transform_array.dot(H), (x_max-x_min, y_max-y_min))


    # 将img2贴到结果贴到原图中
    result_img[transform_dist[1]:transform_dist[1]+h2, 
                transform_dist[0]:transform_dist[0]+w2] = img2

    return result_img