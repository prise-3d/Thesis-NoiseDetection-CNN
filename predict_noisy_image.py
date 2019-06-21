from sklearn.externals import joblib

import numpy as np

from ipfml import processing, utils
from PIL import Image

import sys, os, argparse, json

from keras.models import model_from_json

from modules.utils import config as cfg
from modules.utils import data as dt

from modules.classes.Transformation import Transformation

path                  = cfg.dataset_path
min_max_ext           = cfg.min_max_filename_extension
metric_choices        = cfg.metric_choices_labels
normalization_choices = cfg.normalization_choices

custom_min_max_folder = cfg.min_max_custom_folder

def main():

    # getting all params
    parser = argparse.ArgumentParser(description="Script which detects if an image is noisy or not using specific model")

    parser.add_argument('--image', type=str, help='Image path')
    parser.add_argument('--metrics', type=str, 
                                     help="list of metrics choice in order to compute data",
                                     default='svd_reconstruction, ipca_reconstruction',
                                     required=True)
    parser.add_argument('--params', type=str, 
                                    help="list of specific param for each metric choice (See README.md for further information in 3D mode)", 
                                    default='100, 200 :: 50, 25',
                                    required=True)
    parser.add_argument('--model', type=str, help='.json file of keras model')

    args = parser.parse_args()

    p_img_file   = args.image
    p_metrics    = list(map(str.strip, args.metrics.split(',')))
    p_params     = list(map(str.strip, args.params.split('::')))
    p_model_file = args.model


    with open(p_model_file, 'r') as f:
        json_model = json.load(f)
        model = model_from_json(json_model)
        model.load_weights(p_model_file.replace('.json', '.h5'))

        model.compile(loss='binary_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])

    # load image
    img = Image.open(p_img_file)

    transformations = []

    for id, metric in enumerate(p_metrics):

        if metric not in metric_choices:
            raise ValueError("Unknown metric, please select a correct metric : ", metric_choices)

        transformations.append(Transformation(metric, p_params[id]))

    # getting transformed image
    transformed_images = []

    for transformation in transformations:
        transformed_images.append(transformation.getTransformedImage(img))

    data = np.array(transformed_images)

    # specify the number of dimensions
    img_width, img_height = cfg.keras_img_size
    n_channels = len(transformations)

    if K.image_data_format() == 'channels_first':
        input_shape = (n_channels, img_width, img_height)
    else:
        input_shape = (img_width, img_height, n_channels)

    prediction = model.predict_classes([data])[0][0]

    # output expected from others scripts
    print(prediction)

if __name__== "__main__":
    main()
