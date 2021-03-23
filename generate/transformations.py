# main imports
import os
import numpy as np

# image processing imports
from ipfml.processing import transform, compression
from ipfml.processing import reconstruction
from ipfml.filters import convolution, kernels
from ipfml import utils
import cv2
from skimage.restoration import denoise_nl_means, estimate_sigma

from PIL import Image


def remove_pixel(img, limit):
    
    width, height = img.shape
    
    output = np.zeros((width, height))
    
    for i in range(width):
        for j in range(height):
            
            if img[i,j] <= limit:
                output[i,j] = img[i,j]
                
    return output


def get_random_value(distribution):
    rand = random.uniform(0, 1)
    prob_sum = 0.
    
    for id, prob in enumerate(distribution):
        
        prob_sum += prob
        
        if prob_sum >= rand:
            return id
        
    return len(distribution) - 1


def distribution_from_data(data):
    
    occurences = np.array([data.count(x) for x in set(data)])
    max_occurences = sum(occurences)
    
    return occurences / max_occurences


def fill_image_with_rand_value(img, func, value_to_replace):
    
    width, height = img.shape
    
    output = np.zeros((width, height))
    
    for i in range(width):
        for j in range(height):
            
            if img[i,j] == value_to_replace:
                output[i, j] = func()
            else:
                output[i, j] = img[i, j]
                
    return output

def _compute_relative_error(ref_sv, k_sv):
    ref = np.sqrt(np.sum(np.square(ref_sv)))
    k = np.sqrt(np.sum(np.square(k_sv)))

    return k / ref

def _find_n_components(block, e=0.1):

    s = transform.get_LAB_L_SVD_s(block)
    
    errors = []
    found = False
    k_components = None
    
    for i in range(len(s)):
        
        #Ak = reconstruction.svd(img, [0, i])
        #error = compute_relative_error_matrix(A, Ak)
        error = _compute_relative_error(s, s[i:])
        errors.append(error)
        
        if error < e and not found:
            k_components = (i + 1)
            found = True
            
    return (k_components, errors)

# Transformation class to store transformation method of image and get usefull information
class Transformation():

    def __init__(self, _transformation, _param, _size):
        self.transformation = _transformation
        self.param = _param
        self.size = _size

    def getTransformedImage(self, img):

        if self.transformation == 'svd_reconstruction':
            begin, end = list(map(int, self.param.split(',')))
            h, w = list(map(int, self.size.split(',')))
            img_reconstructed = reconstruction.svd(img, [begin, end])
            data_array = np.array(img_reconstructed, 'uint8')

            img_array = Image.fromarray(data_array)
            img_array.thumbnail((h, w))

            data = np.array(img_array)

        if self.transformation == 'svd_reconstruction':
            begin, end = list(map(int, self.param.split(',')))
            h, w = list(map(int, self.size.split(',')))
            img_reconstructed = reconstruction.svd(img, [begin, end])
            data_array = np.array(img_reconstructed, 'uint8')

            img_array = Image.fromarray(data_array)
            img_array.thumbnail((h, w))

            data = np.array(img_array)

        if self.transformation == 'ipca_reconstruction':
            n_components, batch_size = list(map(int, self.param.split(',')))
            h, w = list(map(int, self.size.split(',')))
            img_reconstructed = reconstruction.ipca(img, n_components, batch_size)
            data_array = np.array(img_reconstructed, 'uint8')
            
            img_array = Image.fromarray(data_array)
            img_array.thumbnail((h, w))

            data = np.array(img_array)

        if self.transformation == 'fast_ica_reconstruction':
            n_components = self.param
            h, w = list(map(int, self.size.split(',')))
            img_reconstructed = reconstruction.fast_ica(img, n_components)
            data_array = np.array(img_reconstructed, 'uint8')
            
            img_array = Image.fromarray(data_array)
            img_array.thumbnail((h, w))

            data = np.array(img_array)

        if self.transformation == 'gini_map':
            # kernel size
            k_w, k_h = list(map(int, self.param.split(',')))
            h, w = list(map(int, self.size.split(',')))

            lab_img = transform.get_LAB_L(img)
            img_mask = convolution.convolution2D(lab_img, kernels.gini, (k_w, k_h))

            # renormalize data
            data_array = np.array(img_mask * 255, 'uint8')

            img_array = Image.fromarray(data_array)
            img_array.thumbnail((h, w))

            data = np.array(img_array)

        if self.transformation == 'sobel_based_filter':
            k_size, p_limit = list(map(int, self.param.split(',')))
            h, w = list(map(int, self.size.split(',')))

            lab_img = transform.get_LAB_L(img)

            weight, height = lab_img.shape

            sobelx = cv2.Sobel(lab_img, cv2.CV_64F, 1, 0, ksize=k_size)
            sobely = cv2.Sobel(lab_img, cv2.CV_64F, 0, 1,ksize=k_size)

            sobel_mag = np.array(np.hypot(sobelx, sobely), 'uint8')  # magnitude
            sobel_mag_limit = remove_pixel(sobel_mag, p_limit)

            # use distribution value of pixel to fill `0` values
            sobel_mag_limit_without_0 = [x for x in sobel_mag_limit.reshape((weight*height)) if x != 0]  
            distribution = distribution_from_data(sobel_mag_limit_without_0)
            min_value = int(min(sobel_mag_limit_without_0))
            l = lambda: get_random_value(distribution) + min_value
            img_reconstructed = fill_image_with_rand_value(sobel_mag_limit, l, 0)
            
            img_reconstructed_norm = utils.normalize_2D_arr(img_reconstructed)
            img_reconstructed_norm = np.array(img_reconstructed_norm*255, 'uint8')
            sobel_reconstructed = Image.fromarray(img_reconstructed_norm)
            sobel_reconstructed.thumbnail((h, w))
        
            data = np.array(sobel_reconstructed)

        if self.transformation == 'nl_mean_noise_mask':
            patch_size, patch_distance = list(map(int, self.param.split(',')))
            h, w = list(map(int, self.size.split(',')))

            img = np.array(img)
            sigma_est = np.mean(estimate_sigma(img, multichannel=True))
    
            patch_kw = dict(patch_size=patch_size,      # 5x5 patches
                            patch_distance=patch_distance,  # 13x13 search area
                            multichannel=True)

            # slow algorithm
            denoise = denoise_nl_means(img, h=0.8 * sigma_est, sigma=sigma_est,
                                    fast_mode=False,
                                    **patch_kw)
            
            denoise = np.array(denoise, 'uint8')
            noise_mask = np.abs(denoise - img)
            
            data_array = np.array(noise_mask, 'uint8')
            
            img_array = Image.fromarray(data_array)
            img_array.thumbnail((h, w))

            data = np.array(img_array)
            
        if self.transformation == 'static':
            # static content, we keep input as it is
            data = img

        return data
    
    def getTransformationPath(self):

        path = self.transformation

        if self.transformation == 'svd_reconstruction':
            begin, end = list(map(int, self.param.split(',')))
            w, h = list(map(int, self.size.split(',')))
            path = os.path.join(path, str(begin) + '_' + str(end) + '_S_' + str(w) + '_' + str(h))
        
        if self.transformation == 'gini_map':
            k_w, k_h = list(map(int, self.param.split(',')))
            w, h = list(map(int, self.size.split(',')))
            path = os.path.join(path, str(k_w) + '_' + str(k_h) + '_S_' + str(w) + '_' + str(h))

        if self.transformation == 'ipca_reconstruction':
            n_components, batch_size = list(map(int, self.param.split(',')))
            w, h = list(map(int, self.size.split(',')))
            path = os.path.join(path, 'N' + str(n_components) + '_' + str(batch_size) + '_S_' + str(w) + '_' + str(h))

        if self.transformation == 'fast_ica_reconstruction':
            n_components = self.param
            w, h = list(map(int, self.size.split(',')))
            path = os.path.join(path, 'N' + str(n_components) + '_S_' + str(w) + '_' + str(h))

        if self.transformation == 'min_diff_filter':
            w_size, h_size, stride = list(map(int, self.param.split(',')))
            w, h = list(map(int, self.size.split(',')))
            path = os.path.join(path, 'W_' + str(w_size) + '_' + str(h_size) + '_Stride_' + str(stride) + '_S_' + str(w) + '_' + str(h))

        if self.transformation == 'sobel_based_filter':
            k_size, p_limit = list(map(int, self.param.split(',')))
            h, w = list(map(int, self.size.split(',')))
            path = os.path.join(path, 'K_' + str(k_size) + '_L' + str(p_limit) + '_S_' + str(w) + '_' + str(h))

        if self.transformation == 'nl_mean_noise_mask':
            patch_size, patch_distance = list(map(int, self.param.split(',')))
            h, w = list(map(int, self.size.split(',')))
            path = os.path.join(path, 'S' + str(patch_size) + '_D' + str(patch_distance) + '_S_' + str(w) + '_' + str(h))

        if self.transformation == 'static':
            # param contains image name to find for each scene
            path = self.param

        return path

    def getName(self):
        return self.transformation

    def getParam(self):
        return self.param

    def __str__( self ):
        return self.transformation + ' transformation with parameter : ' + self.param