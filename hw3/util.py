import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import cv2
import random

categories = ['kitchen', 'store', 'bedroom', 'livingRoom', 'office',
              'industrial', 'suburb', 'insidecity', 'tallbuilding', 'street',
              'highway', 'opencountry', 'coast', 'mountain', 'forest']

def get_image_path(data_path, categories):
    print(f'[INFO] Get paths and labels of train and test datas...')
    train_paths = []
    test_paths = []
    
    train_labels = []
    test_labels = []

    for category in categories:
        for file in os.listdir(data_path + "/train/" + category):
            train_paths.append(data_path + "/train/" + category + '/' + file)
            train_labels.append(category)
        
        for file in os.listdir(data_path + "/test/" + category):
            test_paths.append(data_path + "/test/" + category + '/' + file)
            test_labels.append(category)

    return train_paths, train_labels, test_paths, test_labels

def calculate_accuracy(test_labels, test_predicts, test_paths):
    print(f'[INFO] Calculate the accuracy...')
    n = len(test_labels)
    m = 0
    num = len(categories)
    hit = np.zeros(num)
    total = np.zeros(num)
    accuracy = []

    # record = np.empty((num, 3), dtype=object)
    record = [[[] for _ in range(4)] for _ in range(num)]
    
    for i in range(n):
        idx = categories.index(test_labels[i])
        total[idx] += 1
        record[idx][0].append(test_paths[i])
        if test_labels[i] == test_predicts[i]:
            m = m + 1
            hit[idx] = hit[idx] + 1
            record[idx][1].append(test_paths[i])
        else:
            wrong_idx = categories.index(test_predicts[i])
            record[wrong_idx][2].append(test_paths[i])
            record[idx][3].append(test_paths[i])

    print(f'[INFO] The averagy accuracy is: {m / n}')
    for i in range(num):
        print(f'[INFO] The accuracy of {categories[i]} is: {hit[i]/total[i]}')
        accuracy.append("{:.2%}".format(hit[i]/total[i]))
    accuracy.append("{:.2%}".format(m / n))
    return accuracy, record

def build_confusion_mtx(test_labels_ids, predicted_categories, abbr_categories):
    cm = confusion_matrix(test_labels_ids, predicted_categories)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm_normalized, abbr_categories)
  
def plot_confusion_matrix(cm, category, title='Normalized confusion matrix'):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(category))
    plt.xticks(tick_marks, category, rotation=45)
    plt.yticks(tick_marks, category)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def nest(image_A, image_B):    
    height_A, width_A, _ = image_A.shape
    height_B, width_B, _ = image_B.shape
    x_offset = (width_A - width_B) // 2
    y_offset = (height_A - height_B) // 2
    
    new_image = np.ones((height_A, width_A, 3)) * 255
    
    new_image[y_offset:y_offset+height_B, x_offset:x_offset+width_B] = image_B
    
    return new_image

def write_to_pic(text, height, width):
    blank_image = np.ones((height, width, 3)) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_color = (0, 0, 0)
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    cv2.putText(blank_image, text, (0, text_y), font, font_scale, font_color, thickness)
    return blank_image

def visualization(record, accu):
    h = len(categories)
    w = 5
    
    background = cv2.imread("./pic/background.png")
    corner = cv2.imread("./pic/corner.png")
    sample = cv2.imread("./pic/sample.png")
    true_pic = cv2.imread("./pic/true.png")
    false_pic = cv2.imread("./pic/false.png")
    wrong_pic = cv2.imread("./pic/predictwrong.png")
    accuracy_pic = cv2.imread("./pic/accuracy.png")
    average_pic = cv2.imread("./pic/average.png")
    height = background.shape[0] * h + corner.shape[0] * 2
    width = background.shape[1] * w + corner.shape[1]
    
    result = np.ones((height, width, 3)) * 255
    result[0:corner.shape[0], 0:corner.shape[1]] = corner
    result[0:corner.shape[0], corner.shape[1]:corner.shape[1]+background.shape[1]] = sample
    result[0:corner.shape[0], corner.shape[1]+background.shape[1]:corner.shape[1]+background.shape[1]*2] = true_pic
    result[0:corner.shape[0], corner.shape[1]+background.shape[1]*2:corner.shape[1]+background.shape[1]*3] = false_pic
    result[0:corner.shape[0], corner.shape[1]+background.shape[1]*3:corner.shape[1]+background.shape[1]*4] = wrong_pic
    result[0:corner.shape[0], corner.shape[1]+background.shape[1]*4:corner.shape[1]+background.shape[1]*5] = accuracy_pic
    
    for i in range(h):
        category = categories[i]
        pic = cv2.imread("./pic/"+category+".png")
        result[corner.shape[0]+background.shape[0]*i:corner.shape[0]+background.shape[0]*(i+1), 0:corner.shape[1]] = pic
        for j in range(w-1):
            element = random.choice(record[i][j])
            element_pic = nest(background, cv2.imread(element))
            result[corner.shape[0]+background.shape[0]*i:corner.shape[0]+background.shape[0]*(i+1), corner.shape[1]+background.shape[1]*j:corner.shape[1]+background.shape[1]*(j+1)] = element_pic
        accuacy = write_to_pic(accu[i], background.shape[0], background.shape[1])
        result[corner.shape[0]+background.shape[0]*i:corner.shape[0]+background.shape[0]*(i+1), width-accuracy_pic.shape[1]:width] = accuacy
    
    result[height-corner.shape[0]:height, 0:corner.shape[1]] = average_pic
    average_accuracy = write_to_pic(str(accu[h]), average_pic.shape[0], accuracy_pic.shape[1])
    result[height-corner.shape[0]:height, width-accuracy_pic.shape[1]:width] = average_accuracy
    
    print(f'[INFO] Visualization picture is saved to \".\\visualization.png\"')
    cv2.imwrite("visualization.png",result)