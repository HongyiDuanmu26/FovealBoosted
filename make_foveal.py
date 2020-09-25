import scipy.io as sio
import numpy as np
import cv2
import sys
import os
from PIL import Image
from glob import glob
import math

def genGaussiankernel(width, sigma):
    x = np.arange(-int(width/2), int(width/2)+1, 1, dtype=np.float32)
    x2d, y2d = np.meshgrid(x, x)
    kernel_2d = np.exp(-(x2d ** 2 + y2d ** 2) / (2 * sigma ** 2))
    kernel_2d = kernel_2d / np.sum(kernel_2d)
    return kernel_2d

def pyramid(im, sigma=1, prNum=6):
    height_ori, width_ori, ch = im.shape
    G = im.copy()
    pyramids = [G]
    
    # gaussian blur
    Gaus_kernel2D = genGaussiankernel(5, sigma)
    
    # downsample
    for i in range(1, prNum):
        G = cv2.filter2D(G, -1, Gaus_kernel2D)
        height, width, _ = G.shape
        G = cv2.resize(G, (int(width/2), int(height/2)))
        pyramids.append(G)
    
    
    # upsample
    for i in range(1, 6):
        curr_im = pyramids[i]
        for j in range(i):
            if j < i-1:
                im_size = (curr_im.shape[1]*2, curr_im.shape[0]*2)
            else:
                im_size = (width_ori, height_ori)
            curr_im = cv2.resize(curr_im, im_size)
            curr_im = cv2.filter2D(curr_im, -1, Gaus_kernel2D)
        pyramids[i] = curr_im

    return pyramids




def foveat_img(im, fixs):
    """
    im: input image
    fixs: sequences of fixations of form [(x1, y1), (x2, y2), ...]
    
    This function outputs the foveated image with given input image and fixations.
    """
    sigma=0.248 ### original 0.8 for one cell
    prNum = 6
    As = pyramid(im, sigma, prNum)
    height, width, _ = im.shape
    
    # compute coef
    p = 7.5  ## 7.5

    alpha = 2.5 ## 2.5

    x = np.arange(0, width, 1, dtype=np.float32)
    y = np.arange(0, height, 1, dtype=np.float32)
    x2d, y2d = np.meshgrid(x, y)
    theta = np.sqrt((x2d - fixs[0][0]) ** 2 + (y2d - fixs[0][1]) ** 2) / p
    #theta = np.arctan(theta)
    for fix in fixs[1:]:
        theta = np.minimum(theta, np.sqrt((x2d - fix[0]) ** 2 + (y2d - fix[1]) ** 2) / p)
    R = alpha / (theta + alpha)
    R=(R-np.min(R))/(np.max(R)-np.min(R))
    R=R*(prNum-1)
    
    im_fov = np.zeros_like(As[0], dtype=np.float32)
    
    for i in range(prNum-1):
        ind=np.logical_and(R >= i, R < (i+1))
        print(i,np.sum(ind))
        
        B1=np.abs(R-i)
        B1[~ind]=0
        B2=np.abs(R-(i+1))
        B2[~ind]=0
        
        print(B1[ind])
        
        for j in range(3):
            im_fov[:, :, j] += np.multiply(B2, As[prNum-i-1][:, :, j]) + np.multiply(B1, As[prNum-i-2][:, :, j])
        
    
    im_fov = im_fov.astype(np.uint8)
    return im_fov

if __name__=='__main__':
	imageids=['02','08','14','17','18','19','20','21','23','25','26']
	imagepath='./512image/'
	pointspath='./cor/'
	outputpath1='./foveat/'


	for i in range(len(imageids)):

	    imageid=imageids[i]
	    print('processing',imageid)

	    outputpath1_i=outputpath1+imageid+'/'
	    if not os.path.isdir(outputpath1_i):
	        os.mkdir(outputpath1_i)
	    outputpath2_i=outputpath2+imageid+'/'
	    
	    imagefile=imagepath+imageid+'.tif'
	    image=Image.open(imagefile)
	    image=np.array(image)

	    for pointfile in glob(pointspath+imageid+'/*.mat'):
	        
	        filename=pointfile.split('/')[-1]
	        
	        point=sio.loadmat(pointfile)['cor']
	        x=np.mean(point[:,0]).astype(np.int)
	        y=np.mean(point[:,1]).astype(np.int)
	        im = foveat_img(image,[(x,y)])
	        sio.savemat('{}{}_{:0>3}.mat'.format(outputpath1_i,imageid,filename), {'vol':im})
	        
	    
