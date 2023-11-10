import os
import re
import pickle
import logging
import zipfile
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime
from typing import NoReturn, List, Tuple
from scipy.sparse import dok_matrix

import torch
from torch.utils.data import Dataset, DataLoader


def download_dataset(root_dir, dataset='ml-20m'):
    """
        A helper function to download movielens dataset.
        
        Args:
            root_dir (str): Root directory of the downloaded dataset.
            dataset (str, optional): The name of dataset to be downloaded.
        Outputs:
            The number of users and items in the dataset, and the dataset in pd.DataFrame format.
    """

    # download data if not exists
    # https://files.grouplens.org/datasets/movielens/ml-20m.zip
    if not os.path.exists(root_dir):
        subprocess.call('mkdir ' + root_dir, shell=True)
    if not os.path.exists(os.path.join(root_dir, dataset + '.zip')):
        print('Downloading data into ' + root_dir)
        subprocess.call(
            'cd {} && curl -O http://files.grouplens.org/datasets/movielens/{}.zip'
            .format(root_dir, dataset), shell=True)

        
def preprocess_movielens(root_dir, dataset='ml-20m', random_seed=42):
    """
        A helper function to preprocess the downloaded datasets, and build train/dev/test set in pandas DataFrame.

        Args:
            root_dir (str): Root directory of the downloaded dataset.
            dataset (str, optional): The name of dataset to be downloaded.
        Outputs:
            a dict that contains: n_users, n_items, as well as train, dev, and test sets (in pd.DataFrame format)
    """

    np.random.seed(random_seed)

    # unzip data and read data
    with zipfile.ZipFile(os.path.join(root_dir, dataset + '.zip')) as z:
        with z.open(os.path.join(dataset, 'ratings.csv')) as f:
            data_df = pd.read_csv(f, sep=',')
    data_df = data_df.rename(columns={'userId': 'user_id', 'movieId': 'item_id', 'timestamp': 'time'})

    # statistics of the dataset
    n_users = data_df['user_id'].value_counts().size
    n_items = data_df['item_id'].value_counts().size
    n_clicks = len(data_df)
    min_time = data_df['time'].min()
    max_time = data_df['time'].max()
    print('# Users: ' + str(n_users))
    print('# Items: ' + str(n_items))
    print('# Interactions: ' + str(n_clicks))
    print('Time Span: {}/{}'.format(
        datetime.utcfromtimestamp(min_time).strftime('%Y-%m-%d'),
        datetime.utcfromtimestamp(max_time).strftime('%Y-%m-%d'))
    )

    # drop duplicates and sort
    out_df = data_df.drop_duplicates(['user_id', 'item_id', 'rating', 'time'])
    out_df.sort_values(by=['time', 'user_id'], kind='mergesort', inplace=True)
    out_df = out_df.reset_index(drop=True)

    # reindex
    uids = sorted(out_df['user_id'].unique())
    user2id = dict(zip(uids, range(0, len(uids))))
    iids = sorted(out_df['item_id'].unique())
    item2id = dict(zip(iids, range(0, len(iids))))
    out_df['user_id'] = out_df['user_id'].apply(lambda x: user2id[x])
    out_df['item_id'] = out_df['item_id'].apply(lambda x: item2id[x])

    # split train/dev/test set
    clicked_item_set = dict()
    for user_id, seq_df in out_df.groupby('user_id'):
        clicked_item_set[user_id] = set(seq_df['item_id'].values.tolist())

    def generate_dev_test(data_df, dev_pos_items=5, test_pos_items=5, num_neg_items=1000):
        result_dfs = []
        n_items = data_df['item_id'].value_counts().size
        for set_name in ['test', 'dev']:
            num_pos_items = eval(set_name+"_pos_items")
            result_df = data_df.groupby('user_id', as_index=False).tail(num_pos_items).drop(columns=['time']).copy()
            data_df = data_df.drop(result_df.index)
            result_df = result_df.groupby('user_id', as_index=False).agg({'item_id':lambda x:list(x), 'rating':lambda x:list(x)})
            neg_items = np.random.randint(0, n_items, (len(result_df), num_neg_items))
            for i, uid in enumerate(result_df['user_id'].values):
                user_clicked = clicked_item_set[uid]
                for j in range(len(neg_items[i])):
                    while neg_items[i][j] in user_clicked:
                        neg_items[i][j] = np.random.randint(0, n_items)
            result_df['neg_items'] = neg_items.tolist()
            result_dfs.append(result_df)
        return result_dfs, data_df

    leave_df = out_df.groupby('user_id').head(1)
    data_df = out_df.drop(leave_df.index)

    [test_df, dev_df], data_df = generate_dev_test(data_df)
    train_df = pd.concat([leave_df, data_df]).sort_index()

    data_info = {'n_users': n_users, 'n_items': n_items, 'inters': len(train_df)}

    # save the data_info
    with open(os.path.join(root_dir, 'data_info.pkl'), 'wb') as f:
        pickle.dump(data_info, f)

    # save train_df, dev_df, and text_df in csv files
    train_df.drop(columns='time', inplace=True)
    train_df = train_df.groupby('user_id', as_index=False).agg({'item_id':lambda x: list(x), 'rating':lambda x: list(x)})
    train_df['pos_items'] = train_df['item_id'].apply(lambda x: len(x))      
    train_df.to_csv(os.path.join(root_dir, 'train.csv'), sep='\t', index=False)

    dev_df.to_csv(os.path.join(root_dir, 'dev.csv'), sep='\t', index=False)
    test_df.to_csv(os.path.join(root_dir, 'test.csv'), sep='\t', index=False)

    return data_info


def df_to_dict(df):
    r"""
        Convert the input pd.DataFrame into a dict.
    """
    res = df.to_dict('list')
    for key in res:
        res[key] = np.array(res[key])
    return res

def cal_ideal_dcg(x, topk=-1):
    r"""
        Compute the ideal DCG for a given list.
        
        args:
            x (list): A list of ratings.
            topk (int, optional): If topk=-1, then compute ideal DCG over the full list; 
                    otherwise compute over the topk items of the list.
        Outputs:
                Ideal DCG
    """
    x_sorted = -np.sort(-np.array(x))  # descending order
    pos = np.log2(1.0 + np.arange(1, len(x)+1))
    ideal_dcg_size = topk if topk != -1 else len(x)
    ideal_dcg = np.sum(((2 ** x_sorted - 1) / pos)[:ideal_dcg_size])
    return ideal_dcg


class moivelens_trainset(Dataset):
    """
    The pytorch dataset class for Movielens training sets.

    Args:
        data_path (str): file name, default: 'train.csv'.
        n_users (int): number of users
        n_items (int): number of items
        topk (int, optional): topk value is used to compute the ideal DCG for each user.
    """
    def __init__(self, data_path, n_users, n_items, topk=-1, chunksize=1000):
        super(moivelens_trainset, self).__init__()
        self.data_path = data_path
        self.n_users = n_users
        self.n_items = n_items            
        self.topk = topk

        # prepare self.targets for datasampler, prepare id mapper for loss functions
        self.targets = dok_matrix((n_users, n_items), dtype=np.float32)
        self.id_mapper = dok_matrix((n_users+1, n_items+1), dtype=np.int32)
        id_start = 0

        with pd.read_csv(data_path, sep='\t', chunksize=chunksize) as reader:
            for chunk in reader:
                for idx, row in chunk.iterrows():
                    item_ids, ratings = eval(row['item_id']), eval(row['rating'])
                    self.targets[row['user_id'], item_ids] = ratings
                    self.id_mapper[row['user_id'], item_ids] = np.arange(id_start, id_start+len(item_ids))
                    id_start += len(item_ids)

        self.total_relevant_pairs = id_start 
        self.raw_content = pd.read_csv(self.data_path, sep='\t') #.iloc[0] # skiprows=user_id+1,
        #print(self.raw_content.shape)
        self.targets = self.targets.tocsr()
        self.id_mapper = self.id_mapper.tocsr()

    def __len__(self):
        return self.n_users
    
    def get_num_televant_pairs(self):
        return self.total_relevant_pairs

    def __getitem__(self, index):
        item_ids, user_id = index

        # convert user_id x item_ids to user_item_id
        user_id_repeat = [user_id] * len(item_ids)
        user_item_ids = np.squeeze(np.asarray(self.id_mapper[user_id_repeat, item_ids.tolist()], dtype=np.int_))

        feed_dict = {
            'user_id': user_id,
            'item_id': item_ids,
            'user_item_id': user_item_ids,
            'rating': np.squeeze(self.targets[user_id, item_ids].toarray()), 
            'num_pos_items': self.raw_content['pos_items'][user_id], #int(row_content['pos_items']),
            'ideal_dcg': cal_ideal_dcg(eval(self.raw_content['rating'][user_id]), self.topk)
        }
        return feed_dict

    # Collate a batch according to the list of feed dicts
    def collate_batch(self, feed_dicts):
        feed_dict = dict()
        for key in feed_dicts[0]:
            stack_val = np.array([d[key] for d in feed_dicts])
            feed_dict[key] = torch.from_numpy(stack_val)

        feed_dict['batch_size'] = len(feed_dicts)
        feed_dict['phase'] = 'train'
        return feed_dict

    
class moivelens_evalset(Dataset):
    """
        The pytorch dataset class for Movielens dev/test sets.
        
        Args:
            data_path (str): file name, 'dev.csv' or 'test.csv'
            n_users (int): number of users
            n_items (int): number of items
            phase (string): 'dev' or 'test'
    """
    def __init__(self, data_path, n_users, n_items, phase):
        super(moivelens_evalset, self).__init__()
        self.data_path = data_path
        self.n_users = n_users
        self.n_items = n_items
        self.phase = phase
        self.targets = None

    def __len__(self):
        return self.n_users

    def get_batch(self, index, batchsize):
        contents = pd.read_csv(self.data_path, sep='\t', skiprows=index+1, nrows=batchsize, names=['user_id','item_id','rating','neg_items'])
        user_id = contents['user_id'].astype(int).tolist()
        item_id = contents['item_id'].apply(eval).tolist()
        rating = contents['rating'].apply(eval).tolist()
        neg_items = contents['neg_items'].apply(eval).tolist()

        batch_size = len(user_id)
        item_ids = np.concatenate([item_id, neg_items], axis=1)

        feed_dict = {
            'user_id': torch.tensor(user_id),
            'item_id': torch.tensor(item_ids),
            'rating': torch.tensor(rating),
            'batch_size': batch_size,
            'phase': self.phase
        }
        return feed_dict


class MoiveLens(Dataset):
      r"""A wrapper of MoiveLens dataset. 
      """
      def __init__(self, root, phase='train', topk=-1, random_seed=123):
          
          if os.path.isfile(os.path.join(root, 'ml-20m.zip')):
             print('Files already downloaded and verified')
          else:
             print ('Prepare to download dataset...')
             download_dataset(root_dir=root)

          if os.path.isfile(os.path.join(root, 'data_info.pkl')) and \
             os.path.isfile(os.path.join(root, 'train.csv')) and \
             os.path.isfile(os.path.join(root, 'dev.csv')) and \
             os.path.isfile(os.path.join(root, 'test.csv')):

             with open(os.path.join(root, 'data_info.pkl'), 'rb') as f:
                  self.data_info = pickle.load(f) 
                
             print('# Users: ' + str(self.data_info['n_users']))
             print('# Items: ' + str(self.data_info['n_items']))
             print('# Interactions: ' + str(self.data_info['inters']))
            
          else:
             self.data_info = preprocess_movielens(root_dir=root, random_seed=random_seed)
             print('# Users: ' + str(self.data_info['n_users']))
             print('# Items: ' + str(self.data_info['n_items']))
             print('# Interactions: ' + str(self.data_info['inters']))
  
          self.n_users = self.data_info['n_users']           # number of users in the dataset
          self.n_items = self.data_info['n_items']           # number of items in the dataset         
                          
          if phase == 'train':
             dataset = moivelens_trainset(data_path=os.path.join(root, 'train.csv'), n_users=self.n_users, n_items=self.n_items, topk=topk)
          else:                  
             dataset = moivelens_evalset(data_path=os.path.join(root, phase+'.csv'), n_users=self.n_users, n_items=self.n_items, phase=phase) 

          self.dataset = dataset
          self.phase = phase
          self.targets = dataset.targets

      def get_num_televant_pairs(self):
          assert self.phase == 'train'  
          return self.dataset.get_num_televant_pairs()

      def collate_batch(self, feed_dicts):
          return self.dataset.collate_batch(feed_dicts)

      def get_batch(self, index: int, batchsize: int):
          assert self.phase in ['dev', 'test']
          return self.dataset.get_batch(index, batchsize)        
        
      def __len__(self):
          return self.dataset.__len__()

      def __getitem__(self, index):
          assert self.phase == 'train'
          return self.dataset.__getitem__(index)
    