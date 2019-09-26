# main imports
import sys, os, argparse, json
import numpy as np

# image processing imports
from ipfml import processing, utils
from PIL import Image

# model imports
from sklearn.externals import joblib
from keras.models import model_from_json
from keras import backend as K

# modules imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
from modules.utils import data as dt
from modules.classes.Transformation import Transformation

# parameters from config
path                  = cfg.dataset_path
min_max_ext           = cfg.min_max_filename_extension
features_choices      = cfg.features_choices_labels

custom_min_max_folder = cfg.min_max_custom_folder

def main():

    # getting all params
    parser = argparse.ArgumentParser(description="Script which detects if an image is noisy or not using specific model")

    parser.add_argument('--image', type=str, help='Image path')
    parser.add_argument('--features', type=str, 
                                     help="list of features choice in order to compute data",
                                     default='svd_reconstruction, ipca_reconstruction',
                                     required=True)
    parser.add_argument('--params', type=str, 
                                    help="list of specific param for each feature choice (See README.md for further information in 3D mode)", 
                                    default='100, 200 :: 50, 25',
                                    required=True)
    parser.add_argument('--size', type=str, help="Expected output size before processing transformation", default="100,100")
    parser.add_argument('--model', type=str, help='.json file of keras model')

    args = parser.parse_args()

    p_img_file   = args.image
    p_features   = list(map(str.strip, args.features.split(',')))
    p_params     = list(map(str.strip, args.params.split('::')))
    p_size       = args.size
    p_model_file = args.model


    with open(p_model_file, 'r') as f:
        json_model = json.load(f)
        model = model_from_json(json_model)
        model.load_weights(p_model_file.replace('.json', '.h5'))

        model.compile(loss='binary_crossentropy',
                    optimizer='rmsprop',
                    features=['accuracy'])

    # load image
    img = Image.open(p_img_file)

    transformations = []

    for id, feature in enumerate(p_features):

        if feature not in features_choices:
            raise ValueError("Unknown feature, please select a correct feature : ", features_choices)

        transformations.append(Transformation(feature, p_params[id], p_size))

    # getting transformed image
    transformed_images = []
    
    for transformation in transformations:
        transformed_images.append(transformation.getTransformedImage(img))

    data = np.array(transformed_images)

    # specify the number of dimensions
    img_width, img_height = cfg.sub_image_size
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
