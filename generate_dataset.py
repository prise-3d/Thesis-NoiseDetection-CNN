#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 21:02:42 2018

@author: jbuisine
"""

from __future__ import print_function
import glob, image_slicer
import sys, os, getopt
from PIL import Image
import shutil

# show to create own dataset https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d

NUMBER_SUB_IMAGES = 100

def init_directory():

    if os.path.exists('data'):
        print("Removing all previous data...")

        shutil.rmtree('data')

    if not os.path.exists('data'):
        print("Creating new data...")
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

    global NUMBER_SUB_IMAGES

    if len(sys.argv) <= 1:
        print('Please specify nb sub image per image parameter (use -h if you want to know values)...')
        print('generate_dataset.py --nb xxxx')
        sys.exit(2)
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h:n", ["help", "nb="])
    except getopt.GetoptError:
        # print help information and exit:
        print('generate_dataset.py --nb xxxx')
        sys.exit(2)
    for o, a in opts:

        if o == "-h":
            print('generate_dataset.py --nb xxxx')
            print('20x20 : 1600')
            print('40x40 : 400')
            print('60x60 : 178 (approximately)')
            print('80x80 : 100')
            print('100x100 : 64')
            sys.exit()
        elif o == '--nb':
            NUMBER_SUB_IMAGES = int(a)

    init_directory()

    # create database using img folder (generate first time only)
    generate_dataset()

if __name__== "__main__":
    main()
