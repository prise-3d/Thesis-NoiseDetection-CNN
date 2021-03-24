#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 11:47:42 2019

@author: jbuisine
"""

# main imports
import sys, os, argparse
import numpy as np
import random

# images processing imports
from PIL import Image
from ipfml.processing.segmentation import divide_in_blocks

# modules imports
sys.path.insert(0, '') # trick to enable import of main folder module

import config  as cfg
from transformations import Transformation

def generate_data_model(_filename, _transformations, _dataset_folder, _selected_zones, _sequence):

    output_train_filename = os.path.join(output_data_folder, _filename, _filename + ".train")
    output_test_filename = os.path.join(output_data_folder, _filename, _filename + ".test")

    # create path if not exists
    if not os.path.exists(os.path.join(output_data_folder, _filename)):
        os.makedirs(os.path.join(output_data_folder, _filename))

    train_file = open(output_train_filename, 'w')
    test_file = open(output_test_filename, 'w')

    # train_file_data = []
    # test_file_data  = []

    # specific number of zones (zones indices)
    zones = np.arange(16)

    # go ahead each scenes
    for folder_scene in _selected_zones:

        scene_path = os.path.join(_dataset_folder, folder_scene)

        train_zones = _selected_zones[folder_scene]

        for id_zone, index_folder in enumerate(zones):

            index_str = str(index_folder)
            if len(index_str) < 2:
                index_str = "0" + index_str
            
            current_zone_folder = "zone" + index_str
            zone_path = os.path.join(scene_path, current_zone_folder)

            # custom path for interval of reconstruction and metric

            features_path = []

            for transformation in _transformations:
                
                # check if it's a static content and create augmented images if necessary
                if transformation.getName() == 'static':
                    
                    # {sceneName}/zoneXX/static
                    static_metric_path = os.path.join(zone_path, transformation.getName())

                    # img.png
                    image_name = transformation.getParam().split('/')[-1]

                    # {sceneName}/zoneXX/static/img
                    image_prefix_name = image_name.replace('.png', '')
                    image_folder_path = os.path.join(static_metric_path, image_prefix_name)
                    
                    if not os.path.exists(image_folder_path):
                        os.makedirs(image_folder_path)

                    features_path.append(image_folder_path)

                    # get image path to manage
                    # {sceneName}/static/img.png
                    transform_image_path = os.path.join(scene_path, transformation.getName(), image_name) 
                    static_transform_image = Image.open(transform_image_path)

                    static_transform_image_block = divide_in_blocks(static_transform_image, cfg.sub_image_size)[id_zone]

                    dt.augmented_data_image(static_transform_image_block, image_folder_path, image_prefix_name)

                else:
                    metric_interval_path = os.path.join(zone_path, transformation.getTransformationPath())
                    features_path.append(metric_interval_path)

            # as labels are same for each metric
            for label in os.listdir(features_path[0]):

                label_features_path = []

                for path in features_path:
                    label_path = os.path.join(path, label)
                    label_features_path.append(label_path)

                # getting images list for each metric
                features_images_list = []
                    
                for index_metric, label_path in enumerate(label_features_path):

                    if _transformations[index_metric].getName() == 'static':
                        # by default append nothing..
                        features_images_list.append([])
                    else:
                        images = sorted(os.listdir(label_path))
                        features_images_list.append(images)

                sequence_data = []

                # construct each line using all images path of each
                for index_image in range(0, len(features_images_list[0])):
                    
                    images_path = []

                    # get information about rotation and flip from first transformation (need to be a not static transformation)
                    current_post_fix =  features_images_list[0][index_image].split(cfg.post_image_name_separator)[-1]

                    # getting images with same index and hence name for each metric (transformation)
                    for index_metric in range(0, len(features_path)):

                        # custom behavior for static transformation (need to check specific image)
                        if _transformations[index_metric].getName() == 'static':
                            # add static path with selecting correct data augmented image
                            image_name = _transformations[index_metric].getParam().split('/')[-1].replace('.png', '')
                            img_path = os.path.join(features_path[index_metric], image_name + cfg.post_image_name_separator + current_post_fix)
                            images_path.append(img_path)
                        else:
                            img_path = features_images_list[index_metric][index_image]
                            images_path.append(os.path.join(label_features_path[index_metric], img_path))

                    if label == cfg.noisy_folder:
                        line = '1;'
                    else:
                        line = '0;'

                    # add new data information into sequence
                    sequence_data.append(images_path)

                    if len(sequence_data) >= _sequence:
                        
                        # prepare whole line for LSTM model kind
                        # keeping last noisy label

                        for id_seq, seq_images_path in enumerate(sequence_data):
                            # compute line information with all images paths
                            for id_path, img_path in enumerate(seq_images_path):
                                if id_path < len(seq_images_path) - 1:
                                    line = line + img_path + '::'
                                else:
                                    line = line + img_path

                            if id_seq < len(sequence_data) - 1:
                                line += ';'
                        
                        line = line + '\n'

                        if id_zone in train_zones:
                            # train_file_data.append(line)
                            train_file.write(line)
                        else:
                            # test_file_data.append(line)
                            test_file.write(line)

                        # remove first element (sliding window)
                        del sequence_data[0]

    # random.shuffle(train_file_data)
    # random.shuffle(test_file_data)

    # for line in train_file_data:
    #     train_file.write(line)

    # for line in test_file_data:
    #     test_file.write(line)

    train_file.close()
    test_file.close()

def main():

    parser = argparse.ArgumentParser(description="Compute specific dataset for model using of metric")

    parser.add_argument('--output', type=str, help='output file name desired (.train and .test)')
    parser.add_argument('--folder', type=str,
                    help='folder where generated data are available',
                    required=True) 
    parser.add_argument('--features', type=str,
                                     help="list of features choice in order to compute data",
                                     default='svd_reconstruction, ipca_reconstruction',
                                     required=True)
    parser.add_argument('--params', type=str, 
                                    help="list of specific param for each metric choice (See README.md for further information in 3D mode)", 
                                    default='100, 200 :: 50, 25',
                                    required=True)
    parser.add_argument('--sequence', type=int, help='sequence length expected', required=True)
    parser.add_argument('--size', type=str, 
                                  help="Size of input images",
                                  default="100, 100")
    parser.add_argument('--selected_zones', type=str, help='file which contains all selected zones of scene', required=True)    

    args = parser.parse_args()

    p_filename   = args.output
    p_folder     = args.folder
    p_features   = list(map(str.strip, args.features.split(',')))
    p_params     = list(map(str.strip, args.params.split('::')))
    p_sequence   = args.sequence
    p_size       = args.size # not necessary to split here
    p_selected_zones = args.selected_zones

    selected_zones = {}
    with(open(p_selected_zones, 'r')) as f:

        for line in f.readlines():

            data = line.split(';')
            del data[-1]
            scene_name = data[0]
            thresholds = data[1:]

            selected_zones[scene_name] = [ int(t) for t in thresholds ]

    # create list of Transformation
    transformations = []

    for id, feature in enumerate(p_features):

        if feature not in cfg.features_choices_labels:
            raise ValueError("Unknown metric, please select a correct metric : ", cfg.features_choices_labels)

        transformations.append(Transformation(feature, p_params[id], p_size))

    if transformations[0].getName() == 'static':
        raise ValueError("The first transformation in list cannot be static")


    # create database using img folder (generate first time only)
    generate_data_model(p_filename, transformations, p_folder, selected_zones, p_sequence)

if __name__== "__main__":
    main()
