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

def load_data(data_path, MIL_flag=True):
    tmp = np.load(data_path, allow_pickle=True) # replace this with an url, file size: 8.8 MB.
    Y = tmp['Y']
    if MIL_flag == False:
      X = tmp['oriX']
      X = np.expand_dims(X,axis=1)
    else:
      X = tmp['X']
    X = np.transpose(X,[0,1,4,2,3])
    N = Y.shape[0]
    ids = np.random.permutation(N)
    trN = int(0.9 * N)
    tr_ids = ids[:trN]
    te_ids = ids[trN:]
    train_X = X[tr_ids]
    test_X = X[te_ids]
    train_Y = Y[tr_ids]
    test_Y = Y[te_ids]
    print(train_X.shape)
    print(train_Y.shape)
    print(test_X.shape)
    print(test_Y.shape)
    return  train_X, train_Y, test_X, test_Y


def BreastCancer(root='./data/', MIL_flag=True):
    r"""
        The breast cancer histopathology data from [1]_. The original images can be downloaded at [2]_.

        Args:
            flag(bool, required): whether to use data in multiple instance learning format or not, default: False.

        Example:
            >>> (train_data, train_labels), (test_data, test_labels) = BreastCancer(flag=False)

        Reference:
            .. [1] Gelasca, Elisa Drelie, et al. "Evaluation and benchmark for biological image segmentation." 
               2008 15th IEEE international conference on image processing. IEEE, 2008.
               
            .. [2] https://www.kaggle.com/datasets/andrewmvd/breast-cancer-cell-segmentation?resource=download
    """
    base_folder = "Breast_Cancer"
    url = 'https://github.com/DixianZhu/MIDAM/releases/download/pre-release/breast.npz'
    filename = "breast.npz"
    train_list = [
                  ['breast.npz', None],
                 ]
    test_list = []

    # download dataset 
    if not _check_integrity(root, train_list, test_list, base_folder):
       download_url(url=url, root=os.path.join(root, base_folder), filename=filename)
    
    data_path = os.path.join(root, base_folder, train_list[0][0])
    train_data, train_labels, test_data, test_labels = load_data(data_path, MIL_flag=MIL_flag)
    
    return (train_data, train_labels), (test_data, test_labels)

