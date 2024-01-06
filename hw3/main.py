import TinyImage as tiny
import BagOfSIFT as bag
import util
import classifier
import argparse
import sys
import os
import pickle
import time

data_path = "./data"
categories = ['kitchen', 'store', 'bedroom', 'livingRoom', 'office',
              'industrial', 'suburb', 'insidecity', 'tallbuilding', 'street',
              'highway', 'opencountry', 'coast', 'mountain', 'forest']
CATE2ID = {v: k for k, v in enumerate(categories)}

if __name__ == "__main__":
    start = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--representation", help="method of image representation", dest="representation", type=str, default="tiny")
    parser.add_argument("-s", "--size", help="size of vocabulary", dest="size", type=int, default=200)
    parser.add_argument("-c", "--create", help="create new pretrain model", dest="create", type=bool, default=False)
    parser.add_argument("-v", "--visualization", help="visualize the result", dest="visual", type=bool, default=False)
    args = parser.parse_args()
    
    representation = args.representation
    vocab_size = args.size
    recreate = args.create
    visual = args.visual
    
    train_paths, train_labels, test_paths, test_labels = util.get_image_path(data_path, categories)
    print(f'[INFO] Size of train dataset is: ', len(train_paths))
    print(f'[INFO] Size of test dataset is: ', len(test_paths))
    
    if representation == 'tiny':
        train_images = tiny.get_tiny_image(train_paths)
        test_images = tiny.get_tiny_image(test_paths)
    elif representation == 'bag':
        if not os.path.exists('vocab.pm') or recreate:
            print(f'[INFO] Create visual word vocabulary now...')
            if os.path.exists('vocab.pm'):
                os.remove('vocab.pm')
            vocab = bag.build_vocabulary(train_paths, vocab_size)
            with open('vocab.pm', 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        if not os.path.exists('train.pm') or recreate:
            print(f'[INFO] Create train image feat now...')
            if os.path.exists('train.pm'):
                os.remove('train.pm')
            train_images = bag.get_bags_of_words(train_paths)
            with open('train.pm', 'wb') as handle:
                pickle.dump(train_images, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('train.pm', 'rb') as handle:
                train_images = pickle.load(handle)

        if not os.path.exists('test.pm') or recreate:
            print(f'[INFO] Create test image feat now.')
            if os.path.exists('test.pm'):
                os.remove('test.pm')
            test_images = bag.get_bags_of_words(test_paths)
            with open('test.pm', 'wb') as handle:
                pickle.dump(test_images, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('test.pm', 'rb') as handle:
                test_images = pickle.load(handle)
    else:
        print('[ERROR] Unsupported representation')
        sys.exit(1)
    
    test_predicts = classifier.knn(train_images, train_labels, test_images)
    accuracy, records = util.calculate_accuracy(test_labels, test_predicts, test_paths)

    end = time.time()
    print(f'[INFO] It takes ', end-start, ' seconds')
    
    test_labels_ids = [CATE2ID[x] for x in test_labels]
    predicted_categories_ids = [CATE2ID[x] for x in test_predicts]
    train_labels_ids = [CATE2ID[x] for x in train_labels]

    if visual:
        util.build_confusion_mtx(test_labels_ids, predicted_categories_ids, categories)
        util.visualization(records, accuracy)