import numpy as np
from scipy.spatial.distance import cdist

def knn(train_images, train_labels, test_images):
    print(f'[INFO] Start classify using knn...')
    k = 6
    # distances = np.sqrt(np.sum((train_images[:, np.newaxis] - test_images)**2, axis=2))
    distances = cdist(test_images, train_images, metric='euclidean')
    # distances = cdist(train_images, test_images, metric='euclidean')
    test_predicts = []
    
    for dis in distances:
        knn_idx = np.argsort(dis)
        knn_label = np.array(train_labels)[knn_idx[:k]]
        
        label_counts = {}
        for label in knn_label:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1 
        best_label = max(label_counts, key=label_counts.get)
        test_predicts.append(best_label)

    return test_predicts