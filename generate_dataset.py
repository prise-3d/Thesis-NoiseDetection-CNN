#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 11:47:42 2019

@author: jbuisine
"""

import sys, os, argparse
import numpy as np
import random
import time
import json

from PIL import Image
from ipfml import processing, metrics, utils
from skimage import color

from modules.utils import config as cfg
from modules.utils import data as dt

from preprocessing_functions import svd_reconstruction

# getting configuration information
config_filename         = cfg.config_filename
zone_folder             = cfg.zone_folder
learned_folder          = cfg.learned_zones_folder
min_max_filename        = cfg.min_max_filename_extension

# define all scenes values
scenes_list             = cfg.scenes_names
scenes_indexes          = cfg.scenes_indices
choices                 = cfg.normalization_choices
path                    = cfg.dataset_path
zones                   = cfg.zones_indices
seuil_expe_filename     = cfg.seuil_expe_filename

metric_choices          = cfg.metric_choices_labels
output_data_folder      = cfg.output_data_folder

generic_output_file_svd = '_random.csv'

def generate_data_model(_scenes_list, _filename, _interval,  _metric, _scenes, _nb_zones = 4, _random=0):

    output_train_filename = _filename + ".train"
    output_test_filename = _filename + ".test"

    if not '/' in output_train_filename:
        raise Exception("Please select filename with directory path to save data. Example : data/dataset")

    # create path if not exists
    if not os.path.exists(output_data_folder):
        os.makedirs(output_data_folder)

    train_file_data = []
    test_file_data  = []

    scenes = os.listdir(path)
    # remove min max file from scenes folder
    scenes = [s for s in scenes if min_max_filename not in s]
    begin, end = _interval

    # go ahead each scenes
    for id_scene, folder_scene in enumerate(_scenes_list):

        scene_path = os.path.join(path, folder_scene)

        zones_indices = zones

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

            # custom path for metric
            metric_path = os.path.join(zone_path, _metric)

            # custom path for interval of reconstruction and metric
            metric_interval_path = os.path.join(metric_path, str(begin) + "_" + str(end))

            for label in os.listdir(metric_interval_path):
                label_path = os.path.join(metric_interval_path, label)

                images = os.listdir(label_path)

                for img in images:
                    img_path = os.path.join(label_path, img)

                    if label == cfg.noisy_folder:
                        line = '1;' + img_path + '\n'
                    else:
                        line = '0;' + img_path + '\n'

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
    parser.add_argument('--metric', type=str, 
                                    help="metric choice in order to compute data (use 'all' if all metrics are needed)", 
                                    choices=metric_choices,
                                    required=True)
    parser.add_argument('--interval', type=str, help="interval choice if needed by the compression method", default='"100, 200"')
    parser.add_argument('--scenes', type=str, help='List of scenes to use for training data')
    parser.add_argument('--nb_zones', type=int, help='Number of zones to use for training data set', choices=list(range(1, 17)))
    parser.add_argument('--renderer', type=str, help='Renderer choice in order to limit scenes used', choices=cfg.renderer_choices, default='all')
    parser.add_argument('--random', type=int, help='Data will be randomly filled or not', choices=[0, 1])

    args = parser.parse_args()

    p_filename = args.output
    p_metric   = args.metric
    p_interval = list(map(int, args.interval.split(',')))
    p_scenes   = args.scenes.split(',')
    p_nb_zones = args.nb_zones
    p_renderer = args.renderer
    p_random   = args.random

        # list all possibles choices of renderer
    scenes_list = dt.get_renderer_scenes_names(p_renderer)
    scenes_indices = dt.get_renderer_scenes_indices(p_renderer)

    # getting scenes from indexes user selection
    scenes_selected = []

    for scene_id in p_scenes:
        index = scenes_indices.index(scene_id.strip())
        scenes_selected.append(scenes_list[index])

    # create database using img folder (generate first time only)
    generate_data_model(scenes_list, p_filename, p_interval,  p_metric, scenes_selected, p_nb_zones, p_random)


if __name__== "__main__":
    main()
