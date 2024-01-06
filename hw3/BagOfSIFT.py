import cv2
import numpy as np
import pickle
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from tqdm import tqdm

def build_vocabulary(image_paths, vocab_size=300):
    print(f'[INFO] Size of vocabulary: {vocab_size}')
    image_list = [cv2.imread(file) for file in image_paths]
    sift = cv2.SIFT_create()
    descriptors = []
    for image in tqdm(image_list):
        _, des = sift.detectAndCompute(image, None)
        descriptors.append(des)

    descriptors = np.concatenate(descriptors, axis=0)
    kmeans = KMeans(n_clusters=vocab_size, n_init="auto")
    kmeans.fit(descriptors)
    vocab = kmeans.cluster_centers_

    return vocab

def get_bags_of_words(image_paths):
    with open('vocab.pm', 'rb') as handle:
        vocab = pickle.load(handle)
    
    sift = cv2.SIFT_create()
    image_feats = []
    
    for image_path in tqdm(image_paths):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, des = sift.detectAndCompute(image, None)
        dis = cdist(vocab, des, metric='euclidean')
        idx = np.argmin(dis, axis=0)
        hist, bin_edges = np.histogram(idx, bins=len(vocab))
        hist_norm = [float(i)/sum(hist) for i in hist]
        image_feats.append(hist_norm)
    
    image_feats = np.asarray(image_feats)
    
    return image_feats