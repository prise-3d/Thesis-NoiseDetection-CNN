# main imports
import numpy as np
import pandas as pd
import sys, os, argparse
import json

# model imports
import cnn_models as models
import tensorflow as tf
import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# image processing imports
import cv2
from sklearn.utils import shuffle

# config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg


def main():

    parser = argparse.ArgumentParser(description="Train Keras model and save it into .json file")

    parser.add_argument('--data', type=str, help='dataset filename prefix (without .train and .test)', required=True)
    parser.add_argument('--model', type=str, help='.json file of keras model')

    args = parser.parse_args()

    p_data_file   = args.data
    p_model_file  = args.model
        
    ########################
    # 1. Get and prepare data
    ########################
    print("Preparing data...")
    dataset = pd.read_csv(p_data_file, header=None, sep=";")

    print("Dataset size : ", len(dataset))

    # default first shuffle of data
    dataset = shuffle(dataset)

    print("Reading all images data...")

    # getting number of chanel
    n_channels = len(dataset[1][1].split('::'))
    print("Number of channels : ", n_channels)

    img_width, img_height = cfg.keras_img_size

    # specify the number of dimensions
    if K.image_data_format() == 'channels_first':
        if n_channels > 1:
            input_shape = (1, n_channels, img_width, img_height)
        else:
            input_shape = (n_channels, img_width, img_height)

    else:
        if n_channels > 1:
            input_shape = (1, img_width, img_height, n_channels)
        else:
            input_shape = (img_width, img_height, n_channels)

    # `:` is the separator used for getting each img path
    if n_channels > 1:
        dataset[1] = dataset[1].apply(lambda x: [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in x.split('::')])
    else:
        dataset[1] = dataset[1].apply(lambda x: cv2.imread(x, cv2.IMREAD_GRAYSCALE))

    # reshape array data
    dataset[1] = dataset[1].apply(lambda x: np.array(x).reshape(input_shape))

    # use of the whole data set for training
    x_dataset = dataset.ix[:,1:]
    y_dataset = dataset.ix[:,0]

    x_data = []
    for item in x_dataset.values:
        #print("Item is here", item)
        x_data.append(item[0])

    x_data = np.array(x_data)

    print("End of loading data..")

    #######################
    # 2. Getting model
    #######################

    with open(p_model_file, 'r') as f:
        json_model = json.load(f)
        model = model_from_json(json_model)
        model.load_weights(p_model_file.replace('.json', '.h5'))

        model.compile(loss='binary_crossentropy',
                    optimizer='rmsprop',
                    features=['accuracy'])


    # Get results obtained from model
    y_data_prediction = model.predict(x_data)

    y_prediction = [1 if x > 0.5 else 0 for x in y_data_prediction]

    acc_score = accuracy_score(y_dataset, y_prediction)
    f1_data_score = f1_score(y_dataset, y_prediction)
    recall_data_score = recall_score(y_dataset, y_prediction)
    pres_score = precision_score(y_dataset, y_prediction)
    roc_score = roc_auc_score(y_dataset, y_prediction)

    # save model performance
    if not os.path.exists(cfg.results_information_folder):
        os.makedirs(cfg.results_information_folder)

    perf_file_path = os.path.join(cfg.results_information_folder, cfg.perf_prediction_model_path)

    # write header if necessary
    if not os.path.exists(perf_file_path):
        with open(perf_file_path, 'w') as f:
            f.write(cfg.perf_prediction_header_file)

    # add information into file
    with open(perf_file_path, 'a') as f:
        line = p_data_file + ';' + p_model_file + ';' + str(acc_score) + ';' + str(f1_data_score) + ';' + str(recall_data_score) + ';' + str(pres_score) + ';' + str(roc_score)
        f.write(line)

if __name__== "__main__":
    main()
