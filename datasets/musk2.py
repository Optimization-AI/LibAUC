import os
import os.path
import numpy as np
from torchvision.datasets.utils import check_integrity, download_url

def _check_integrity(root, train_list, test_list, base_folder):
    for fentry in (train_list + test_list):
        filename, md5 = fentry[0], fentry[1]
        fpath = os.path.join(root, base_folder, filename)
        if not check_integrity(fpath, md5):
          return False
    print('Files already downloaded and verified')
    return True

def load_data(data_path):
    tmp = np.load(data_path, allow_pickle=True) # replace this with an url, file size: 8.8 MB.
    train_data = tmp['train_X']
    test_data = tmp['test_X']
    train_labels = tmp['train_Y'].astype(int)
    test_labels = tmp['test_Y'].astype(int)
    return train_data, train_labels, test_data, test_labels


def MUSK2(root='./data/'):
    base_folder = "MUSK2"
    url = 'https://github.com/DixianZhu/MIDAM/releases/download/pre-release/musk_2.npz'
    filename = "musk_2.npz"
    train_list = [
                  ['musk_2.npz', None],
                 ]
    test_list = []

    # download dataset 
    if not _check_integrity(root, train_list, test_list, base_folder):
       download_url(url=url, root=os.path.join(root, base_folder), filename=filename)
    
    data_path = os.path.join(root, base_folder, train_list[0][0])
    train_data, train_labels, test_data, test_labels = load_data(data_path)
    
    return (train_data, train_labels), (test_data, test_labels)