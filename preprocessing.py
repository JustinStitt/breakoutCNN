import numpy as np

def to_grayscale(img):
    gray =  np.mean(img, axis = 2).astype(np.uint8)#remove color
    gray = gray.astype('float32')#normalize
    gray /= 255. #normalize
    return gray

def downsample(img):
    return img[::2,::2]#reduce by factor of 2

def preprocess(img):
    _img =  to_grayscale(downsample(img))#perform both preprocessing steps and return new img
    _img.shape = (_img.shape[0],_img.shape[1],1)
    return _img
