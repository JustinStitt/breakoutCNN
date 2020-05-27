import numpy as np

def to_grayscale(img):
    return np.mean(img, axis = 2).astype(np.uint8)

def downsample(img):
    return img[::2,::2]#reduce by factor of 2

def preprocess(img):
    return to_grayscale(downsample(img))#perform both preprocessing steps and return new img
