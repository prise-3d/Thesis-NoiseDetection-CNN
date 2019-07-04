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

import custom_config  as cfg
from modules.utils import data as dt
from modules.classes.Transformation import Transformation

# getting configuration information
config_filename         = cfg.config_filename
zone_folder             = cfg.zone_folder
learned_folder          = cfg.learned_zones_folder
min_max_filename        = cfg.min_max_filename_extension

# define all scenes values
scenes_list             = cfg.scenes_names
scenes_indexes          = cfg.scenes_indices
dataset_path            = cfg.dataset_path
zones                   = cfg.zones_indices
seuil_expe_filename     = cfg.seuil_expe_filename

features_choices        = cfg.features_choices_labels
output_data_folder      = cfg.output_data_folder

generic_output_file_svd = '_random.csv'

def generate_data_model(_scenes_list, _filename, _transformations, _scenes, _nb_zones = 4, _random=0):

    output_train_filename = _filename + ".train"
    output_test_filename = _filename + ".test"

    if not '/' in output_train_filename:
        raise Exception("Please select filename with directory path to save data. Example : data/dataset")

    # create path if not exists
    if not os.path.exists(output_data_folder):
        os.makedirs(output_data_folder)

    zones_indices = zones

    train_file_data = []
    test_file_data  = []

    scenes = os.listdir(dataset_path)
    # remove min max file from scenes folder
    scenes = [s for s in scenes if min_max_filename not in s]

    # go ahead each scenes
    for folder_scene in _scenes_list:

        scene_path = os.path.join(dataset_path, folder_scene)

        # shuffle list of zones (=> randomly choose zones)
        # only in random mode
        if _random:
            random.shuffle(zones_indices)

         # store zones learned
        learned_zones_indices = zones_indices[:_nb_zones]

        # write into file
        folder_learned_path = os.path.join(learned_folder, _filename.split('/')[1])

        if not os.path.exists(folder_learned_path):
            os.makedirs(folder_learned_path)

        file_learned_path = os.path.join(folder_learned_path, folder_scene + '.csv')

        with open(file_learned_path, 'w') as f:
            for i in learned_zones_indices:
                f.write(str(i) + ';')

        for id_zone, index_folder in enumerate(zones_indices):

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

                    static_transform_image_block = divide_in_blocks(static_transform_image, cfg.keras_img_size)[id_zone]

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

                    # compute line information with all images paths
                    for id_path, img_path in enumerate(images_path):
                        if id_path < len(images_path) - 1:
                            line = line + img_path + '::'
                        else:
                            line = line + img_path
                    
                    line = line + '\n'

                    if id_zone < _nb_zones and folder_scene in _scenes:
                        train_file_data.append(line)
                    else:
                        test_file_data.append(line)

    train_file = open(output_train_filename, 'w')
    test_file = open(output_test_filename, 'w')

    random.shuffle(train_file_data)
    random.shuffle(test_file_data)

    for line in train_file_data:
        train_file.write(line)

    for line in test_file_data:
        test_file.write(line)

    train_file.close()
    test_file.close()

def main():

    parser = argparse.ArgumentParser(description="Compute specific dataset for model using of metric")

    parser.add_argument('--output', type=str, help='output file name desired (.train and .test)')
    parser.add_argument('--features', type=str, 
                                     help="list of features choice in order to compute data",
                                     default='svd_reconstruction, ipca_reconstruction',
                                     required=True)
    parser.add_argument('--params', type=str, 
                                    help="list of specific param for each metric choice (See README.md for further information in 3D mode)", 
                                    default='100, 200 :: 50, 25',
                                    required=True)
    parser.add_argument('--scenes', type=str, help='List of scenes to use for training data')
    parser.add_argument('--nb_zones', type=int, help='Number of zones to use for training data set', choices=list(range(1, 17)))
    parser.add_argument('--renderer', type=str, help='Renderer choice in order to limit scenes used', choices=cfg.renderer_choices, default='all')
    parser.add_argument('--random', type=int, help='Data will be randomly filled or not', choices=[0, 1])

    args = parser.parse_args()

    p_filename = args.output
    p_features  = list(map(str.strip, args.features.split(',')))
    p_params   = list(map(str.strip, args.params.split('::')))
    p_scenes   = args.scenes.split(',')
    p_nb_zones = args.nb_zones
    p_renderer = args.renderer
    p_random   = args.random

    # create list of Transformation
    transformations = []

    for id, feature in enumerate(p_features):

        if feature not in features_choices:
            raise ValueError("Unknown metric, please select a correct metric : ", features_choices)

        transformations.append(Transformation(feature, p_params[id]))

    if transformations[0].getName() == 'static':
        raise ValueError("The first transformation in list cannot be static")

    # list all possibles choices of renderer
    scenes_list = dt.get_renderer_scenes_names(p_renderer)
    scenes_indices = dt.get_renderer_scenes_indices(p_renderer)

    # getting scenes from indexes user selection
    scenes_selected = []

    for scene_id in p_scenes:
        index = scenes_indices.index(scene_id.strip())
        scenes_selected.append(scenes_list[index])

    # create database using img folder (generate first time only)
    generate_data_model(scenes_list, p_filename, transformations, scenes_selected, p_nb_zones, p_random)

if __name__== "__main__":
    main()
