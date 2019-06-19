from numpy.linalg import svd
from PIL import Image
from scipy import misc

import time

import numpy as np
from ipfml import metrics

def svd_reconstruction(img, interval):
    
    begin, end = interval
    lab_img = metrics.get_LAB_L(img)
    lab_img = np.array(lab_img, 'uint8')
    
    U, s, V = svd(lab_img, full_matrices=True)
    
    # reconstruction using specific interval
    smat = np.zeros((end-begin, end-begin), dtype=complex)
    smat[:, :] = np.diag(s[begin:end])
    output_img = np.dot(U[:, begin:end],  np.dot(smat, V[begin:end, :]))
        
    return output_img