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
from transformations import Transformation

# modules imports
sys.path.insert(0, '') # trick to enable import of main folder module

import config as cfg
zones = np.arange(16)

def get_scene_image_quality(img_path):

    # if path getting last element (image name) and extract quality
    img_postfix = img_path.split('/')[-1].split(cfg.scene_image_quality_separator)[-1]
    img_quality = img_postfix.replace(cfg.scene_image_extension, '')

    return int(img_quality)

'''
Display progress information as progress bar
'''
def write_progress(progress):
    barWidth = 180

    output_str = "["
    pos = barWidth * progress
    for i in range(barWidth):
        if i < pos:
           output_str = output_str + "="
        elif i == pos:
           output_str = output_str + ">"
        else:
            output_str = output_str + " "

    output_str = output_str + "] " + str(int(progress * 100.0)) + " %\r"
    print(output_str)
    sys.stdout.write("\033[F")


def generate_data(transformation, _dataset_path, _output, _human_thresholds, _replace):
    """
    @brief Method which generates all .csv files from scenes
    @return nothing
    """

    # path is the default dataset path
    scenes = os.listdir(_dataset_path)
    n_scenes = len(scenes)

    # go ahead each scenes
    for id_scene, folder_scene in enumerate(scenes):

        print('Scene {0} of {1} ({2})'.format((id_scene + 1), n_scenes, folder_scene))
        scene_path = os.path.join(_dataset_path, folder_scene)
        output_scene_path = os.path.join(cfg.output_data_generated, _output, folder_scene)

        # construct each zones folder name
        zones_folder = []
        features_folder = []

        if folder_scene in _human_thresholds:

            zones_threshold = _human_thresholds[folder_scene]
            # get zones list info
            for index in zones:
                index_str = str(index)
                if len(index_str) < 2:
                    index_str = "0" + index_str

                current_zone = "zone"+index_str
                zones_folder.append(current_zone)
                zone_path = os.path.join(output_scene_path, current_zone)

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
                    # Disable use of data augmentation for the moment
                    # Data augmentation!
                    # rotations = [0, 90, 180, 270]

                    #img_flip_labels = ['original', 'horizontal', 'vertical', 'both']
                    # img_flip_labels = ['original', 'horizontal']

                    # output_images_path = []
                    # check_path_exists = []
                    # # rotate and flip image to increase dataset size
                    # for id, flip_label in enumerate(img_flip_labels):
                    #     for rotation in rotations:
                    #         output_reconstructed_filename = img_path.split('/')[-1].replace('.png', '') + '_' + zones_folder[id_block] + cfg.post_image_name_separator
                    #         output_reconstructed_filename = output_reconstructed_filename + flip_label + '_' + str(rotation) + '.png'
                    #         output_reconstructed_path = os.path.join(label_path, output_reconstructed_filename)

                    #         if os.path.exists(output_reconstructed_path):
                    #             check_path_exists.append(True)
                    #         else:
                    #             check_path_exists.append(False)

                    #         output_images_path.append(output_reconstructed_path)

                    # compute only if not exists or necessary to replace
                    # if _replace or not np.array(check_path_exists).all():
                        # compute image
                        # pass block to grey level
                        # output_block = transformation.getTransformedImage(block)
                        # output_block = np.array(output_block, 'uint8')
                        
                        # # current output image
                        # output_block_img = Image.fromarray(output_block)

                        #horizontal_img = output_block_img.transpose(Image.FLIP_LEFT_RIGHT)
                        #vertical_img = output_block_img.transpose(Image.FLIP_TOP_BOTTOM)
                        #both_img = output_block_img.transpose(Image.TRANSPOSE)

                        #flip_images = [output_block_img, horizontal_img, vertical_img, both_img]
                        #flip_images = [output_block_img, horizontal_img]

                        # Only current image img currenlty
                        # flip_images = [output_block_img]

                        # # rotate and flip image to increase dataset size
                        # counter_index = 0 # get current path index
                        # for id, flip in enumerate(flip_images):
                        #     for rotation in rotations:

                        #         if _replace or not check_path_exists[counter_index]:
                        #             rotated_output_img = flip.rotate(rotation)
                        #             rotated_output_img.save(output_images_path[counter_index])

                        #         counter_index +=1
                    
                    if _replace:
                        
                        _, filename = os.path.split(img_path)

                        # build of output image filename
                        filename = filename.replace('.png', '')
                        filename_parts = filename.split('_')

                        # get samples : `00XXX`
                        n_samples = filename_parts[-1]
                        del filename_parts[-1]

                        # `p3d_XXXXXX`
                        output_reconstructed = '_'.join(filename_parts)

                        output_reconstructed_filename = output_reconstructed + '_' + zones_folder[id_block] + '_' + n_samples + '.png'
                        output_reconstructed_path = os.path.join(label_path, output_reconstructed_filename)

                        output_block = transformation.getTransformedImage(block)
                        output_block = np.array(output_block, 'uint8')
                        
                        # current output image
                        output_block_img = Image.fromarray(output_block)
                        output_block_img.save(output_reconstructed_path)


                write_progress((id_img + 1) / number_scene_image)

            print('\n')

    print("{0}_{1} : end of data generation\n".format(transformation.getName(), transformation.getParam()))


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
    parser.add_argument('--folder', type=str,
                            help='folder where dataset is available',
                            required=True)  
    parser.add_argument('--output', type=str,
                                help='output folder where data are saved',
                                required=True)              
    parser.add_argument('--thresholds', type=str, help='file which cantains all thresholds', required=True)                  
    parser.add_argument('--size', type=str, 
                                help="specific size of image", 
                                default='100, 100',
                                required=True)
    parser.add_argument('--replace', type=int, help='replace previous picutre', default=1)

    args = parser.parse_args()

    p_features   = list(map(str.strip, args.features.split(',')))
    p_params     = list(map(str.strip, args.params.split('::')))
    p_folder     = args.folder
    p_output     = args.output
    p_thresholds = args.thresholds
    p_size       = args.size
    p_replace    = bool(args.replace)

    # list of transformations
    transformations = []

    for id, feature in enumerate(p_features):

        if feature not in cfg.features_choices_labels or feature == 'static':
            raise ValueError("Unknown feature {0}, please select a correct feature (`static` excluded) : {1}".format(feature, cfg.features_choices_labels))
        
        transformations.append(Transformation(feature, p_params[id], p_size))

    human_thresholds = {}

    # 3. retrieve human_thresholds
    # construct zones folder
    with open(p_thresholds) as f:
        thresholds_line = f.readlines()

        for line in thresholds_line:
            data = line.split(';')
            del data[-1] # remove unused last element `\n`
            current_scene = data[0]
            thresholds_scene = data[1:]

            if current_scene != '50_shades_of_grey':
                human_thresholds[current_scene] = [ int(threshold) for threshold in  thresholds_scene ]


    # generate all or specific feature data
    for transformation in transformations:
        generate_data(transformation, p_folder, p_output, human_thresholds, p_replace)

if __name__== "__main__":
    main()
