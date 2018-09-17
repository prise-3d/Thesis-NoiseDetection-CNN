#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 21:02:42 2018

@author: jbuisine
"""

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os, glob, image_slicer
from PIL import Image

# show to create own dataset https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d

NUMBER_SUB_IMAGES = 1600

def init_directory():
    if not os.path.exists('data'):

        os.makedirs('data/train/final')
        os.makedirs('data/train/noisy')

        os.makedirs('data/validation/final')
        os.makedirs('data/validation/noisy')

def create_images(folder, output_folder):
    images_path = glob.glob(folder + "/*.png")

    for img in images_path:
        image_name = img.replace(folder, '').replace('/', '')
        tiles = image_slicer.slice(img, NUMBER_SUB_IMAGES, save = False)
        image_slicer.save_tiles(tiles, directory=output_folder, prefix='part_'+image_name)

def generate_dataset():
    create_images('img_train/final', 'data/train/final')
    create_images('img_train/noisy', 'data/train/noisy')
    create_images('img_validation/final', 'data/validation/final')
    create_images('img_validation/noisy', 'data/validation/noisy')

def main():

    init_directory()

    # create database using img folder (generate first time only)
    generate_dataset()

if __name__== "__main__":
    main()
