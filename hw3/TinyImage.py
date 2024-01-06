import cv2
import numpy as np

def get_tiny_image(image_paths):
    print(f'[INFO] Get tiny images...')
    size = 16   # recommend is 16 x 16
    tiny_images = []
    
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (size, size))
        image = image.flatten()
        
        image = (image - np.mean(image)) / np.std(image)
        tiny_images.append(image)

    tiny_images = np.asarray(tiny_images)
    return tiny_images