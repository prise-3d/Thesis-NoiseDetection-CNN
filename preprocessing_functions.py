from numpy.linalg import svd
from PIL import Image
import matplotlib.pyplot as plt
from scipy import misc

import time

import numpy as np
from sklearn import preprocessing
import ipfml as iml

def get_s_model_data(image):

    s = iml.metrics.get_SVD_s(image)
    size = len(s)

    # normalized output
    output_normalized = preprocessing.normalize(s, norm='l1', axis=0, copy=True, return_norm=False)

    result = output_normalized.reshape([size, 1, 3])

    return result

def get_s_model_data_img(image, ):
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 1
    fig_size[1] = 1
    plt.rcParams["figure.figsize"] = fig_size

    s = iml.metrics.get_SVD_s(image)

    plt.figure()   # create a new figure

    output_normalized = preprocessing.normalize(s, norm='l1', axis=0, copy=True, return_norm=False)
    plt.plot(output_normalized[70:100, 0])
    plt.plot(output_normalized[70:100:, 1])
    plt.plot(output_normalized[70:100:, 2])

    img = iml.image_processing.fig2img(plt.gcf())

    plt.close('all')

    return img