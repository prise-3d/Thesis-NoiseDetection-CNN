# module file which contains all image metrics used in project

from numpy.linalg import svd
from PIL import Image
import matplotlib.pyplot as plt
from scipy import misc

import time

import numpy as np
from sklearn import preprocessing

import modules.model_helper.image_conversion as img_c

'''
Method which extracts SVD features from image and returns 's' vector
@return 's' vector
'''
def get_s_model_data(image):
    U, s, V = svd(image, full_matrices=False)
    size = len(s)

    # normalized output
    output_normalized = preprocessing.normalize(s, norm='l1', axis=0, copy=True, return_norm=False)

    result = output_normalized.reshape([size, 1, 3])

    return result

def get_s_model_data_img(image):
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 1
    fig_size[1] = 1
    plt.rcParams["figure.figsize"] = fig_size

    U, s, V = svd(image, full_matrices=False)

    plt.figure()   # create a new figure

    output_normalized = preprocessing.normalize(s, norm='l1', axis=0, copy=True, return_norm=False)
    plt.plot(output_normalized[70:100, 0])
    plt.plot(output_normalized[70:100:, 1])
    plt.plot(output_normalized[70:100:, 2])

    img = img_c.fig2img(plt.gcf())

    plt.close('all')

    return img

def get(image):
    return svd(image, full_matrices=False)

def get_s(image):
    U, s, V = svd(image, full_matrices=False)
    return s

def get_U(image):
    U, s, V = svd(image, full_matrices=False)
    return U

def get_V(image):
    U, s, V = svd(image, full_matrices=False)
    return V
