# module file which contains all image metrics used in project

from numpy.linalg import svd
from PIL import Image
from scipy import misc

'''
Method which extracts SVD features from image and returns 's' vector
@return 's' vector
'''
def get_s_model_data(image):
    U, s, V = svd(image, full_matrices=False)
    size = len(s)
    result = s.reshape([size, 1, 1]) # one shape per canal
    return result

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
