#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 11:47:42 2019

@author: jbuisine
"""

# main imports
import sys, os, argparse
import numpy as np

# images processing imports
from PIL import Image
from ipfml.processing.segmentation import divide_in_blocks

# modules imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
from modules.utils.data import get_scene_image_quality
from modules.classes.Transformation import Transformation

# getting configuration information
zone_folder             = cfg.zone_folder
min_max_filename        = cfg.min_max_filename_extension

# define all scenes values
scenes_list             = cfg.scenes_names
scenes_indices          = cfg.scenes_indices
path                    = cfg.dataset_path
zones                   = cfg.zones_indices
seuil_expe_filename     = cfg.seuil_expe_filename

features_choices        = cfg.features_choices_labels
output_data_folder      = cfg.output_data_folder

generic_output_file_svd = '_random.csv'

def generate_data(transformation, _scenes, _replace):
    """
    @brief Method which generates all .csv files from scenes
    @return nothing
    """

    scenes = os.listdir(path)
    # remove min max file from scenes folder
    scenes = [s for s in scenes if min_max_filename not in s]

    # go ahead each scenes
    for id_scene, folder_scene in enumerate(scenes):

        if folder_scene in _scenes:
            print(folder_scene)
            scene_path = os.path.join(path, folder_scene)

            # construct each zones folder name
            zones_folder = []
            features_folder = []
            zones_threshold = []

            # get zones list info
            for index in zones:
                index_str = str(index)
                if len(index_str) < 2:
                    index_str = "0" + index_str

                current_zone = "zone"+index_str
                zones_folder.append(current_zone)
                zone_path = os.path.join(scene_path, current_zone)

                with open(os.path.join(zone_path, cfg.seuil_expe_filename)) as f:
                    zones_threshold.append(int(f.readline()))

                # custom path for feature
                feature_path = os.path.join(zone_path, transformation.getName())

                if not os.path.exists(feature_path):
                    os.makedirs(feature_path)

                # custom path for interval of reconstruction and feature
                feature_interval_path = os.path.join(zone_path, transformation.getTransformationPath())
                features_folder.append(feature_interval_path)

                if not os.path.exists(feature_interval_path):
                    os.makedirs(feature_interval_path)

                # create for each zone the labels folder
                labels = [cfg.not_noisy_folder, cfg.noisy_folder]

                for label in labels:
                    label_folder = os.path.join(feature_interval_path, label)

                    if not os.path.exists(label_folder):
                        os.makedirs(label_folder)

            # get all images of folder
            scene_images = sorted([os.path.join(scene_path, img) for img in os.listdir(scene_path) if cfg.scene_image_extension in img])
            number_scene_image = len(scene_images)

            # for each images
            for id_img, img_path in enumerate(scene_images):

                current_img = Image.open(img_path)
                img_blocks = divide_in_blocks(current_img, cfg.sub_image_size)

                current_quality_index = int(get_scene_image_quality(img_path))

                for id_block, block in enumerate(img_blocks):

                    ##########################
                    # Image computation part #
                    ##########################

                    label_path = features_folder[id_block]

                    # get label folder for block
                    if current_quality_index > zones_threshold[id_block]:
                        label_path = os.path.join(label_path, cfg.not_noisy_folder)
                    else:
                        label_path = os.path.join(label_path, cfg.noisy_folder)

                    # check if necessary to compute or not images
                    # Data augmentation!
                    rotations = [0, 90, 180, 270]

                    #img_flip_labels = ['original', 'horizontal', 'vertical', 'both']
                    img_flip_labels = ['original', 'horizontal']

                    output_images_path = []
                    check_path_exists = []
                    # rotate and flip image to increase dataset size
                    for id, flip_label in enumerate(img_flip_labels):
                        for rotation in rotations:
                            output_reconstructed_filename = img_path.split('/')[-1].replace('.png', '') + '_' + zones_folder[id_block] + cfg.post_image_name_separator
                            output_reconstructed_filename = output_reconstructed_filename + flip_label + '_' + str(rotation) + '.png'
                            output_reconstructed_path = os.path.join(label_path, output_reconstructed_filename)

                            if os.path.exists(output_reconstructed_path):
                                check_path_exists.append(True)
                            else:
                                check_path_exists.append(False)

                            output_images_path.append(output_reconstructed_path)

                    # compute only if not exists or necessary to replace
                    if _replace or not np.array(check_path_exists).all():
                        # compute image
                        # pass block to grey level
                        output_block = transformation.getTransformedImage(block)
                        output_block = np.array(output_block, 'uint8')
                        
                        # current output image
                        output_block_img = Image.fromarray(output_block)

                        horizontal_img = output_block_img.transpose(Image.FLIP_LEFT_RIGHT)
                        #vertical_img = output_block_img.transpose(Image.FLIP_TOP_BOTTOM)
                        #both_img = output_block_img.transpose(Image.TRANSPOSE)

                        #flip_images = [output_block_img, horizontal_img, vertical_img, both_img]
                        flip_images = [output_block_img, horizontal_img]

                        # rotate and flip image to increase dataset size
                        counter_index = 0 # get current path index
                        for id, flip in enumerate(flip_images):
                            for rotation in rotations:

                                if _replace or not check_path_exists[counter_index]:
                                    rotated_output_img = flip.rotate(rotation)
                                    rotated_output_img.save(output_images_path[counter_index])

                                counter_index +=1

                print(transformation.getName() + "_" + folder_scene + " - " + "{0:.2f}".format(((id_img + 1) / number_scene_image)* 100.) + "%")
                sys.stdout.write("\033[F")

            print('\n')

    print("%s_%s : end of data generation\n" % (transformation.getName(), transformation.getParam()))


def main():

    parser = argparse.ArgumentParser(description="Compute and prepare data of feature of all scenes using specific interval if necessary")

    parser.add_argument('--features', type=str, 
                                     help="list of features choice in order to compute data",
                                     default='svd_reconstruction, ipca_reconstruction',
                                     required=True)
    parser.add_argument('--params', type=str, 
                                    help="list of specific param for each feature choice (See README.md for further information in 3D mode)", 
                                    default='100, 200 :: 50, 25',
                                    required=True)
    parser.add_argument('--size', type=str, 
                                help="specific size of image", 
                                default='100, 100',
                                required=True)
    parser.add_argument('--scenes', type=str, help='List of scenes to use for training data')
    parser.add_argument('--replace', type=int, help='replace previous picutre', default=1)

    args = parser.parse_args()

    p_features  = list(map(str.strip, args.features.split(',')))
    p_params    = list(map(str.strip, args.params.split('::')))
    p_size      = args.size
    p_scenes    = args.scenes.split(',')
    p_replace   = bool(args.replace)

    # getting scenes from indexes user selection
    scenes_selected = []

    for scene_id in p_scenes:
        index = scenes_indices.index(scene_id.strip())
        scenes_selected.append(scenes_list[index])

    # list of transformations
    transformations = []

    for id, feature in enumerate(p_features):

        if feature not in features_choices or feature == 'static':
            raise ValueError("Unknown feature, please select a correct feature (`static` excluded) : ", features_choices)

        transformations.append(Transformation(feature, p_params[id], p_size))

    print("Scenes used", scenes_selected)
    # generate all or specific feature data
    for transformation in transformations:
        generate_data(transformation, scenes_selected, p_replace)

if __name__== "__main__":
    main()
