import torch
import skimage.io 
import scipy.io as sio
from torch.utils.data.dataset import Dataset
import os
import random
import numpy as np
import torchvision.transforms as transforms
import h5py
import tifffile as tf
import hdf5storage


class LoadDataset(Dataset):
    def __init__(self, imageFolder1, imageFolder2):
        self.imageFolder1 = imageFolder1
        self.imageFolder2 = imageFolder2
        self.path1 = imageFolder1
        self.path2 = imageFolder2
        self.dataname1 = os.listdir(self.path1)
        self.dataname2 = os.listdir(self.path2)
        

    def __getitem__(self, Index):
        self.dataFile2 = self.imageFolder2+self.dataname2[Index]
        self.img2=h5py.File(self.dataFile2,'r')
        self.img2 = self.img2['rad'][:]/1.0
        self.img2 = self.img2.astype(float)
        #self.img2 = sio.loadmat(self.dataFile2)
        #self.img2 = self.img2['data']/1.0
        #self.img2 = tf.imread(self.dataFile2)/1.0
        #self.img2 = hdf5storage.loadmat(self.dataFile2)
        #self.img2 = self.img2['data']/1.0
        self.img2 = torch.from_numpy(self.img2)
        #self.img2 = self.img2.permute(2,0,1)
        self.img2 = self.img2/torch.max(self.img2)

        self.dataFile1 = self.imageFolder1+self.dataname1[Index%len(self.dataname1)]
        self.img1 = sio.loadmat(self.dataFile1)
        self.img1 = self.img1['guidance_image']/1.0
        self.img1 = torch.from_numpy(self.img1)
        self.img1 = self.img1.permute(2,0,1)
        self.img1 = self.img1/torch.max(self.img1)

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
            ])

        self.img1 = transform(self.img1)
        self.img2 = transform(self.img2)

        self.img = {"guidance": self.img1, "srhsi":self.img2}

        return self.img
    


    def __len__(self):
        return len(self.dataname2)
