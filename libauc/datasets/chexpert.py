import numpy as np
import torch 
from torch.utils.data import Dataset
import torchvision.transforms as tfs
import cv2
from PIL import Image
import pandas as pd

class CheXpert(Dataset):
    r"""
        Reference:
            .. [1] Yuan, Zhuoning, Yan, Yan, Sonka, Milan, and Yang, Tianbao.
               "Large-scale robust deep auc maximization: A new surrogate loss and empirical studies on medical image classification."
               Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.
               https://arxiv.org/abs/2012.03173
    """
    def __init__(self, 
                 csv_path, 
                 image_root_path='',
                 image_size=320,
                 class_index=0, 
                 use_frontal=True,
                 use_upsampling=True,
                 flip_label=False,
                 shuffle=True,
                 seed=123,
                 verbose=False,
                 transforms=None,
                 upsampling_cols=['Cardiomegaly', 'Consolidation'],
                 train_cols=['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis',  'Pleural Effusion'],
                 return_index=False,
                 mode='train'):
        
    
        # load data from csv
        self.df = pd.read_csv(csv_path)
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0-small/', '', regex=True)
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0/', '', regex=True)
        if use_frontal:
            self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']  
            
        # upsample selected cols
        if use_upsampling:
            assert isinstance(upsampling_cols, list), 'Input should be list!'
            sampled_df_list = []
            for col in upsampling_cols:
                print ('Upsampling %s...'%col)
                sampled_df_list.append(self.df[self.df[col] == 1])
            self.df = pd.concat([self.df] + sampled_df_list, axis=0)


        # impute missing values 
        for col in train_cols:
            if col in ['Edema', 'Atelectasis']:
                self.df[col].replace(-1, 1, inplace=True)  
                self.df[col].fillna(0, inplace=True) 
            elif col in ['Cardiomegaly','Consolidation',  'Pleural Effusion']:
                self.df[col].replace(-1, 0, inplace=True) 
                self.df[col].fillna(0, inplace=True)
            elif col in ['No Finding', 'Enlarged Cardiomediastinum', 'Lung Opacity', 'Lung Lesion', 'Pneumonia', 'Pneumothorax', 'Pleural Other','Fracture','Support Devices']: # other labels
                self.df[col].replace(-1, 0, inplace=True) 
                self.df[col].fillna(0, inplace=True)
            else:
                self.df[col].fillna(0, inplace=True)
        
        self._num_images = len(self.df)
        
        # 0 --> -1
        if flip_label and class_index != -1: # In multi-class mode we disable this option!
            self.df.replace(0, -1, inplace=True)   
            
        # shuffle data
        if shuffle:
            data_index = list(range(self._num_images))
            np.random.seed(seed)
            np.random.shuffle(data_index)
            self.df = self.df.iloc[data_index]
        
        
        #assert class_index in [-1, 0, 1, 2, 3, 4], 'Out of selection!'
        assert image_root_path != '', 'You need to pass the correct location for the dataset!'

        if class_index == -1: # 5 classes
            if verbose:
                print ('Multi-label mode: True, Number of classes: [%d]'%len(train_cols))
                print ('-'*30)
            self.select_cols = train_cols
            self.value_counts_dict = {}
            for class_key, select_col in enumerate(train_cols):
                class_value_counts_dict = self.df[select_col].value_counts().to_dict()
                self.value_counts_dict[class_key] = class_value_counts_dict
        else:     
            self.select_cols = [train_cols[class_index]]  # this var determines the number of classes
            self.value_counts_dict = self.df[self.select_cols[0]].value_counts().to_dict()
        
        self.mode = mode
        self.class_index = class_index
        self.image_size = image_size
        self.transforms = transforms
        self.return_index = return_index
        
        self._images_list =  [image_root_path+path for path in self.df['Path'].tolist()]
        if class_index != -1:
            self.targets = self.df[train_cols].values[:, class_index].tolist()
        else:
            self.targets = self.df[train_cols].values.tolist()
    
        if True:
            if class_index != -1:
                if flip_label:
                    self.imratio = self.value_counts_dict[1]/(self.value_counts_dict[-1]+self.value_counts_dict[1])
                    if verbose:
                        print ('-'*30)
                        print('Found %s images in total, %s positive images, %s negative images'%(self._num_images, self.value_counts_dict[1], self.value_counts_dict[-1] ))
                        print ('%s(C%s): imbalance ratio is %.4f'%(self.select_cols[0], class_index, self.imratio ))
                        print ('-'*30)
                else:
                    self.imratio = self.value_counts_dict[1]/(self.value_counts_dict[0]+self.value_counts_dict[1])
                    if verbose:
                        print ('-'*30)
                        print('Found %s images in total, %s positive images, %s negative images'%(self._num_images, self.value_counts_dict[1], self.value_counts_dict[0] ))
                        print ('%s(C%s): imbalance ratio is %.4f'%(self.select_cols[0], class_index, self.imratio ))
                        print ('-'*30)
            else:
                imratio_list = []
                for class_key, select_col in enumerate(train_cols):
                    try:
                        imratio = self.value_counts_dict[class_key][1]/(self.value_counts_dict[class_key][0]+self.value_counts_dict[class_key][1])
                    except:
                        if len(self.value_counts_dict[class_key]) == 1 :
                            only_key = list(self.value_counts_dict[class_key].keys())[0]
                            if only_key == 0:
                                self.value_counts_dict[class_key][1] = 0
                                imratio = 0 # no postive samples
                            else:    
                                self.value_counts_dict[class_key][1] = 0
                                imratio = 1 # no negative samples
                            
                    imratio_list.append(imratio)
                    if verbose:
                        #print ('-'*30)
                        print('Found %s images in total, %s positive images, %s negative images'%(self._num_images, self.value_counts_dict[class_key][1], self.value_counts_dict[class_key][0] ))
                        print ('%s(C%s): imbalance ratio is %.4f'%(select_col, class_key, imratio ))
                        print ()
                        #print ('-'*30)
                self.imratio = np.mean(imratio_list)
                self.imratio_list = imratio_list
                
            
    @property        
    def class_counts(self):
        return self.value_counts_dict
    
    @property
    def imbalance_ratio(self):
        return self.imratio

    @property
    def num_classes(self):
        return len(self.select_cols)
       
    @property  
    def data_size(self):
        return self._num_images 
    
    def image_augmentation(self, image):
        img_aug = tfs.Compose([tfs.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05), scale=(0.95, 1.05), fill=128)]) # pytorch 3.7: fillcolor --> fill
        image = img_aug(image)
        return image
    
    def __len__(self):
        return self._num_images
    
    def __getitem__(self, idx):

        image = cv2.imread(self._images_list[idx], 0)
        image = Image.fromarray(image)
        if self.mode == 'train' :
            if self.transforms is None:
                image = self.image_augmentation(image)
            else:
                image = self.transforms(image)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # resize and normalize; e.g., ToTensor()
        image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)  
        image = image/255.0
        __mean__ = np.array([[[0.485, 0.456, 0.406]]])
        __std__ =  np.array([[[0.229, 0.224, 0.225]  ]]) 
        image = (image-__mean__)/__std__
        image = image.transpose((2, 0, 1)).astype(np.float32)
        if self.class_index != -1: # multi-class mode
            label = np.array(self.targets[idx]).reshape(-1).astype(np.float32)
        else:
            label = np.array(self.targets[idx]).reshape(-1).astype(np.float32)
        if self.return_index:
           return image, label, idx   
        return image, label


if __name__ == '__main__':
    root = '../chexpert/dataset/CheXpert-v1.0-small/'
    traindSet = CheXpert(csv_path=root+'train.csv', image_root_path=root, use_upsampling=True, use_frontal=True, image_size=320, mode='train', class_index=0)
    testSet =  CheXpert(csv_path=root+'valid.csv',  image_root_path=root, use_upsampling=False, use_frontal=True, image_size=320, mode='valid', class_index=0)
    trainloader =  torch.utils.data.DataLoader(traindSet, batch_size=32, num_workers=2, drop_last=True, shuffle=True)
    testloader =  torch.utils.data.DataLoader(testSet, batch_size=32, num_workers=2, drop_last=False, shuffle=False)

 
