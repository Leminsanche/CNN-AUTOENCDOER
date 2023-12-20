import os
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import glob
import matplotlib.image as img
import numpy as np

class Presurisation_Data(Dataset):

    def __init__(self, img_dir, ext = '*.png',split = 'train'):
        self.img_dir = img_dir                                  # Direccion de donde estan las imagenes
        self.list_img = glob.glob(os.path.join(img_dir, ext))
        train, Test =  train_test_split(self.list_img, test_size= 0.1, train_size= 0.9,random_state=232)
        Train, Val =  train_test_split(train, test_size= 0.1, train_size= 0.9,random_state=654)
         
        if split == 'train':
            self.list_img = Train

        elif split == 'val':
            self.list_img = Val

        elif split == 'test':
            self.list_img = Test


    def __len__(self):
        return len(self.list_img)


    def __getitem__(self, idx):    #Segun el indice que se entrega retorna el tensor de la imagen y su clasificacion# 
        img_name = self.list_img[idx]
        image = torch.tensor(np.transpose(img.imread(img_name)[:,:,:3], (2,0,1)))
        return image 
    

class Presurisation_Data_NPZ(Dataset):

    def __init__(self, img_dir, ext = '*.npz',split = 'train'):
        self.img_dir = img_dir                                  # Direccion de donde estan las imagenes
        self.list_img = glob.glob(os.path.join(img_dir, ext))
        train, Test =  train_test_split(self.list_img, test_size= 0.1, train_size= 0.9,random_state=232)
        Train, Val =  train_test_split(train, test_size= 0.1, train_size= 0.9,random_state=654)
         
        if split == 'train':
            self.list_img = Train

        elif split == 'val':
            self.list_img = Val

        elif split == 'test':
            self.list_img = Test


    def __len__(self):
        return len(self.list_img)


    def __getitem__(self, idx):    #Segun el indice que se entrega retorna el tensor de la imagen y su clasificacion# 
        img_name = self.list_img[idx]
        image = torch.tensor(np.transpose(np.load(img_name)['arr_0'][:,:,:3], (2,0,1)))
        return image.float()