import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import os


def get_augmentations_v1(image_size=256, is_test=True):
    import albumentations as A
    from albumentations.pytorch.transforms import ToTensor
    '''
    https://www.kaggle.com/vishnus/a-simple-pytorch-starter-code-single-fold-93
    '''
    imagenet_stats = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
    train_tfms = A.Compose([
        A.Cutout(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2,
                                       ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=50,
                val_shift_limit=50)
        ], p=0.5),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.5),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=0.5), 
        ToTensor(normalize=imagenet_stats)
        ])
    
    test_tfms = A.Compose([
        ToTensor(normalize=imagenet_stats)
        ])
    if is_test:
        return test_tfms
    else:
        return train_tfms

class Melanoma(Dataset):
    r'''
        Reference:
           - https://www.kaggle.com/cdeotte/jpeg-melanoma-256x256
           - https://www.kaggle.com/vishnus/a-simple-pytorch-starter-code-single-fold-93
           - https://www.kaggle.com/haqishen/1st-place-soluiton-code-small-ver
    '''
    def __init__(self, root, test_size=0.2, is_test=False, transforms=None):
        assert os.path.isfile(root + '/train.csv'), 'There is no train.csv in %s!'%root
        self.data = pd.read_csv(root + '/train.csv')
        self.train_df, self.test_df = self.get_train_val_split(self.data, test_size=test_size)
        self.is_test = is_test
       
        if is_test:    
            self.df = self.test_df.copy()
        else:
            self.df = self.train_df.copy()
            
        self._num_images = len(self.df)
        self.value_counts_dict = self.df.target.value_counts().to_dict()
        self.imratio = self.value_counts_dict[1]/self.value_counts_dict[0]
        print ('Found %s image in total, %s postive images, %s negative images.'%(self._num_images, self.value_counts_dict[1], self.value_counts_dict[0]))

        # get path 
        dir_name = 'train' 
        self._images_list = [f"{root}/{dir_name}/{img}.jpg" for img in self.df.image_name]
        self._labels_list =  self.df.target.values.tolist()
        if not transforms:
            self.transforms = get_augmentations_v1(is_test=is_test)
        else:
            self.transforms = transforms(is_test=is_test)
            
    @property        
    def class_counts(self):
        return self.value_counts_dict
    
    @property
    def imbalance_ratio(self):
        return self.imratio

    @property
    def num_classes(self):
        return 1
    
    def get_train_val_split(self, df, test_size=0.2):
        print ('test set split is %s'%test_size)
        #Remove Duplicates
        df = df[df.tfrecord != -1].reset_index(drop=True)
        #We are splitting data based on triple stratified kernel provided here https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/165526
        num_tfrecords = len(df.tfrecord.unique())
        train_tf_records = list(range(len(df.tfrecord.unique())))[:-int(num_tfrecords*test_size)]
        split_cond = df.tfrecord.apply(lambda x: x in train_tf_records)
        train_df = df[split_cond].reset_index()
        valid_df = df[~split_cond].reset_index()
        return train_df, valid_df
    
    def __len__(self):
        return self.df.shape[0]   
    
    def __getitem__(self,idx):
        img_path = self._images_list[idx]
        image = Image.open(img_path)
        image = self.transforms(**{"image": np.array(image)})["image"]
        target = torch.tensor([self._labels_list[idx]],dtype=torch.float32) 
        return image, target
    
if __name__ == '__main__':    
    trainSet = Melanoma(root='./datasets/256x256/', is_test=False, test_size=0.2)
    testSet = Melanoma(root='./datasets/256x256/', is_test=True, test_size=0.2)
    bs = 128
    train_dl = DataLoader(dataset=trainSet,batch_size=bs,shuffle=True, num_workers=0)


    