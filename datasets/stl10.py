import os
import os.path
import numpy as np
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, verify_str_arg
# reference: https://pytorch.org/vision/0.8/_modules/torchvision/datasets/stl10.html#STL10

def load_file(data_file, labels_file=None):
    labels = None
    if labels_file:
       with open(labels_file, 'rb') as f:
          labels = np.fromfile(f, dtype=np.uint8) - 1  # 0-based
    with open(data_file, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3, 96, 96))
        images = np.transpose(images, (0, 1, 3, 2))

    return images, labels

def _check_integrity(root, train_list, test_list, base_folder):
    for fentry in (train_list + test_list):
        filename, md5 = fentry[0], fentry[1]
        fpath = os.path.join(root, base_folder, filename)
        if not check_integrity(fpath, md5):
          return False
    print('Files already downloaded and verified')
    return True

def STL10(root='./data/', split='train'):
    base_folder = 'stl10_binary'
    url = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
    filename = "stl10_binary.tar.gz"
    class_names_file = 'class_names.txt'
    folds_list_file = 'fold_indices.txt'
    train_list = [
        ['train_X.bin', '918c2871b30a85fa023e0c44e0bee87f'],
        ['train_y.bin', '5a34089d4802c674881badbb80307741'],
        ['unlabeled_X.bin', '5242ba1fed5e4be9e1e742405eb56ca4']
    ]

    test_list = [
        ['test_X.bin', '7f263ba9f9e0b06b93213547f721ac82'],
        ['test_y.bin', '36f9794fa4beb8a2c72628de14fa638e']
    ]
    splits = ('train', 'train+unlabeled', 'unlabeled', 'test')
  
    # download dataset 
    fpath = os.path.join(root, base_folder, filename)
    if not _check_integrity(root, train_list, test_list, base_folder):
       download_and_extract_archive(url=url, download_root=root, filename=filename)

    # choose which set to load 
    if split=='train':
       path_to_data = os.path.join(root, base_folder, train_list[0][0])
       path_to_labels = os.path.join(root, base_folder, train_list[1][0])
       data, targets = load_file(path_to_data, path_to_labels)
    elif split == 'unlabeled': 
       path_to_data = os.path.join(root, base_folder, train_list[2][0])
       data, _ = load_file(path_to_data)
       targets = np.asarray([-1] * data.shape[0])   
    elif split == 'test': 
       path_to_data = os.path.join(root, base_folder, test_list[0][0])
       path_to_labels = os.path.join(root, base_folder, test_list[1][0])
       data, targets = load_file(path_to_data, path_to_labels)   
    else:
       raise ValueError('Out of option!') 
	
    return data, targets


    
if __name__ == '__main__':   
  data, targets = STL10(root='./data/', split='test') # return numpy array
  print (data.shape, targets.shape)
