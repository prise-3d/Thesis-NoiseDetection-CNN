# main imports
import numpy as np
import pandas as pd
import sys, os, argparse

# image processing
from PIL import Image
from ipfml import utils
from ipfml.processing import transform, segmentation

import matplotlib.pyplot as plt

# model imports
import joblib
from keras.models import load_model
from keras import backend as K

# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
import modules.utils.data as dt
from modules.classes.Transformation import Transformation

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

def main():

    parser = argparse.ArgumentParser(description="Read and compute entropy data file")

    parser.add_argument('--model', type=str, help='model .h5 file')
    parser.add_argument('--folder', type=str,
                        help='folder where scene dataset is available',
                        required=True)  
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
    parser.add_argument('--sequence', type=int, help='sequence size expected', required=True, default=1)
    parser.add_argument('--n_stop', type=int, help='number of detection to make sure to stop', default=1)
    parser.add_argument('--save', type=str, help='filename where to save input data')
    parser.add_argument('--label', type=str, help='label to use when saving thresholds')

    args = parser.parse_args()

    p_model    = args.model
    p_folder   = args.folder
    p_features = list(map(str.strip, args.features.split(',')))
    p_params   = list(map(str.strip, args.params.split('::')))
    p_size     = args.size
    p_sequence = args.sequence
    p_n_stop   = args.n_stop
    p_save     = args.save
    p_label    = args.label

    # 1. Load expected transformations

    # list of transformations
    transformations = []

    for id, feature in enumerate(p_features):

        if feature not in cfg.features_choices_labels or feature == 'static':
            raise ValueError("Unknown feature, please select a correct feature (`static` excluded) : ", cfg.features_choices_labels)

        transformations.append(Transformation(feature, p_params[id], p_size))

    # 2. load model and compile it

    # TODO : check kind of model
    model = joblib.load(p_model)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    # model = load_model(p_model)
    # model.compile(loss='binary_crossentropy',
    #               optimizer='rmsprop',
    #               metrics=['accuracy'])


    estimated_thresholds = []
    n_estimated_thresholds = []
    sequence_list_zones = []

    scene_path = p_folder

    if not os.path.exists(scene_path):
        print('Unvalid scene path:', scene_path)
        exit(0)

    # 3. retrieve human_thresholds
    # construct zones folder
    zones_indices = np.arange(16)
    zones_list = []

    for index in zones_indices:

        index_str = str(index)

        while len(index_str) < 2:
            index_str = "0" + index_str
        
        zones_list.append(cfg.zone_folder + index_str)


    # 4. get estimated thresholds using model and specific method
    images_path = sorted([os.path.join(scene_path, img) for img in os.listdir(scene_path) if cfg.scene_image_extension in img])
    number_of_images = len(images_path)
    image_indices = [ dt.get_scene_image_quality(img_path) for img_path in images_path ]

    image_counter = 0

    # append empty list
    for _ in zones_list:
        estimated_thresholds.append(None)
        n_estimated_thresholds.append(0)
        sequence_list_zones.append([])

    for img_i, img_path in enumerate(images_path):

        blocks = segmentation.divide_in_blocks(Image.open(img_path), (200, 200))

        for index, block in enumerate(blocks):
            
            sequence_list = sequence_list_zones[index]

            if estimated_thresholds[index] is None:
                
                transformed_list = []
                # compute data here
                for transformation in transformations:
                    transformed = transformation.getTransformedImage(block)
                    transformed_list.append(transformed)

                data = np.array(transformed_list)

                sequence_list.append(data)
                
                if len(sequence_list) >= p_sequence:
                    # compute input size
                    # n_chanels, _, _ = data.shape

                    input_data = np.array(sequence_list)
                        
                    input_data = np.expand_dims(input_data, axis=0)

                    prob = model.predict(np.array(input_data))[0]
                    #print(index, ':', image_indices[img_i], '=>', prediction)
                
                    # if prob is now near to label `0` then image is not longer noisy
                    if prob < 0.5:
                        n_estimated_thresholds[index] += 1

                        # if same number of detection is attempted
                        if n_estimated_thresholds[index] >= p_n_stop:
                            estimated_thresholds[index] = image_indices[img_i]
                    else:
                        n_estimated_thresholds[index] = 0

                    # remove first image
                    del sequence_list[0]

        # write progress bar
        write_progress((image_counter + 1) / number_of_images)
        
        image_counter = image_counter + 1
    
    # default label
    for i, _ in enumerate(zones_list):
        if estimated_thresholds[i] == None:
            estimated_thresholds[i] = image_indices[-1]

    # 6. save estimated thresholds into specific file
    print('\nEstimated thresholds', estimated_thresholds)
    if p_save is not None:
        with open(p_save, 'a') as f:
            f.write(p_label + ';')

            for t in estimated_thresholds:
                f.write(str(t) + ';')
            f.write('\n')
    

if __name__== "__main__":
    main()