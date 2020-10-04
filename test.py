from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="2";  
import csv
import pandas as ps
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import random
import cv2
import tensorflow as tf
from datetime import datetime
import sklearn.preprocessing
#import skimage
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import keras
from keras.datasets import cifar10
from keras.regularizers import L1L2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, multiply, Dense, Dropout, Activation, Flatten, GaussianNoise
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ReLU, PReLU, concatenate, AlphaDropout
from keras.activations import selu
from keras import optimizers
from keras import regularizers
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from unet import *
import scipy.io as sio
from glob import glob

def load_imgphi(path,pid):

    files=glob(path+'foveal/'+pid+'/*.mat')
    files.sort()
    numimage=len(files)
    images1=np.zeros((numimage,512,512,3))
    
    phi=np.zeros((numimage,512,512,1))
    for i in range(numimage):
        j=i+1
        filename1=path+'foveal/'+pid+'/'+pid+'_{:0>3}.mat'.format(j)
        
        img=sio.loadmat(filename1)['vol']
        img=cv2.resize(img,(512,512),interpolation=cv2.INTER_CUBIC)
        images1[i,:,:,:]=img.astype(np.float32)/256.0
        

        filename4=path+'cells/'+pid+'/{:0>3}.mat'.format(j)
        phi[i,:,:,0]=sio.loadmat(filename4)['vol'].astype(np.float32)

    return images1,phi






def CustomedLoss(kernel):
    def CustomedLoss_prior(y_true,y_pred):
        BCE=binary_crossentropy(y_true, y_pred)

        prior_shapes=K.constant(kernel)
        prior=K.conv2d(y_pred,prior_shapes,padding='same')
        prior=K.max(prior, axis=[-1])
        

        first=np.array([[1,1,0],[1,0,-1],[0,-1,-1]])
        first=np.reshape(first,(3,3,1,1))
        firstD=K.constant(first)
        firstD_loss=K.conv2d(y_pred,firstD,padding='same')
        firstD_loss=K.mean(K.abs(firstD_loss),axis=[-1])

        second=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
        second=np.reshape(second,(3,3,1,1))
        secondD=K.constant(second)
        secondD_loss=K.conv2d(y_pred,secondD,padding='same')
        secondD_loss=K.mean(K.abs(secondD_loss),axis=[-1])

        
        return BCE+0.5*(526-prior)/526.0 + 0.5*firstD_loss  + 0.05*secondD_loss
    return CustomedLoss_prior


def cnncheck(): 
     
    x_test=load_img('testfile name')
    
    
    model=load_model('./weights/model.h5')
    pred= model.predict(x_test, batch_size=batch_size, verbose=1, steps=None)
    
    sio.savemat('./results/results.mat',{'vol':pred})
    

    return
    

if __name__=='__main__':
    cnncheck()
    K.clear_session()

