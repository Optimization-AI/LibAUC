import os
import os.path
import numpy as np
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
# reference: https://pytorch.org/vision/0.8/_modules/torchvision/datasets/cifar.html#CIFAR10
# Dataset credit goes to https://www.microsoft.com/en-us/download/details.aspx?id=54765

def _check_integrity(root, train_list, test_list, base_folder):
    for fentry in (train_list + test_list):
        filename, md5 = fentry[0], fentry[1]
        fpath = os.path.join(root, base_folder, filename)
        if not check_integrity(fpath, md5):
          return False
    print('Files already downloaded and verified')
    return True

def load_data(data_path, label_path):
    data = np.load(data_path)
    targets = np.load(label_path)
    return data, targets

def CAT_VS_DOG(root='./data/', train=True):
    base_folder = "cat_vs_dog"
    url = 'https://homepage.divms.uiowa.edu/~zhuoning/datasets/cat_vs_dog.tar.gz'
    filename = "cat_vs_dog.tar.gz"
    train_list = [
                  ['cat_vs_dog_data.npy', None],
                  ['cat_vs_dog_label.npy', None],
                  ]
    test_list = []

    # download dataset 
    if not _check_integrity(root, train_list, test_list, base_folder):
       download_and_extract_archive(url=url, download_root=root, filename=filename)
    
    # train or test set
    if train:
       data_path = os.path.join(root, base_folder, train_list[0][0])
       label_path = os.path.join(root, base_folder, train_list[1][0])
       data, targets = load_data(data_path, label_path) 
       data = data[:-5000]
       targets = targets[:-5000]
    else: 
       data_path = os.path.join(root, base_folder, train_list[0][0])
       label_path = os.path.join(root, base_folder, train_list[1][0])
       data, targets = load_data(data_path, label_path) 
       data = data[-5000:]
       targets = targets[-5000:] 

    return data, targets

if __name__ == '__main__':
    data, targets = CAT_VS_DOG('./data/', train=True)
    print (data.shape, targets.shape)
    data, targets = CAT_VS_DOG('./data/', train=False)
    print (data.shape, targets.shape)
