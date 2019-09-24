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
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# image processing imports
import cv2
from sklearn.utils import shuffle

# config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg


def main():

    # default keras configuration
    config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8}) 
    sess = tf.Session(config=config) 
    keras.backend.set_session(sess)

    parser = argparse.ArgumentParser(description="Train Keras model and save it into .json file")

    parser.add_argument('--data', type=str, help='dataset filename prefix (without .train and .test)', required=True)
    parser.add_argument('--output', type=str, help='output file name desired for model (without .json extension)', required=True)
    parser.add_argument('--tl', type=int, help='use or not of transfer learning (`VGG network`)', default=0, choices=[0, 1])
    parser.add_argument('--batch_size', type=int, help='batch size used as model input', default=cfg.keras_batch)
    parser.add_argument('--epochs', type=int, help='number of epochs used for training model', default=cfg.keras_epochs)
    parser.add_argument('--val_size', type=float, help='percent of validation data during training process', default=cfg.val_dataset_size)

    args = parser.parse_args()

    p_data_file  = args.data
    p_output     = args.output
    p_tl         = args.tl
    p_batch_size = args.batch_size
    p_epochs     = args.epochs
    p_val_size   = args.val_size
        
    ########################
    # 1. Get and prepare data
    ########################
    print("Preparing data...")
    dataset_train = pd.read_csv(p_data_file + '.train', header=None, sep=";")
    dataset_test = pd.read_csv(p_data_file + '.test', header=None, sep=";")

    print("Train set size : ", len(dataset_train))
    print("Test set size : ", len(dataset_test))

    # default first shuffle of data
    dataset_train = shuffle(dataset_train)
    dataset_test = shuffle(dataset_test)

    print("Reading all images data...")

    # getting number of chanel
    n_channels = len(dataset_train[1][1].split('::'))
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
        dataset_train[1] = dataset_train[1].apply(lambda x: [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in x.split('::')])
        dataset_test[1] = dataset_test[1].apply(lambda x: [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in x.split('::')])
    else:
        dataset_train[1] = dataset_train[1].apply(lambda x: cv2.imread(x, cv2.IMREAD_GRAYSCALE))
        dataset_test[1] = dataset_test[1].apply(lambda x: cv2.imread(x, cv2.IMREAD_GRAYSCALE))

    # reshape array data
    dataset_train[1] = dataset_train[1].apply(lambda x: np.array(x).reshape(input_shape))
    dataset_test[1] = dataset_test[1].apply(lambda x: np.array(x).reshape(input_shape))

    # get dataset with equal number of classes occurences
    noisy_df_train = dataset_train[dataset_train.ix[:, 0] == 1]
    not_noisy_df_train = dataset_train[dataset_train.ix[:, 0] == 0]
    nb_noisy_train = len(noisy_df_train.index)

    noisy_df_test = dataset_test[dataset_test.ix[:, 0] == 1]
    not_noisy_df_test = dataset_test[dataset_test.ix[:, 0] == 0]
    nb_noisy_test = len(noisy_df_test.index)

    final_df_train = pd.concat([not_noisy_df_train[0:nb_noisy_train], noisy_df_train])
    final_df_test = pd.concat([not_noisy_df_test[0:nb_noisy_test], noisy_df_test])

    # shuffle data another time
    final_df_train = shuffle(final_df_train)
    final_df_test = shuffle(final_df_test)

    final_df_train_size = len(final_df_train.index)
    final_df_test_size = len(final_df_test.index)

    # use of the whole data set for training
    x_dataset_train = final_df_train.ix[:,1:]
    x_dataset_test = final_df_test.ix[:,1:]

    y_dataset_train = final_df_train.ix[:,0]
    y_dataset_test = final_df_test.ix[:,0]

    x_data_train = []
    for item in x_dataset_train.values:
        #print("Item is here", item)
        x_data_train.append(item[0])

    x_data_train = np.array(x_data_train)

    x_data_test = []
    for item in x_dataset_test.values:
        #print("Item is here", item)
        x_data_test.append(item[0])

    x_data_test = np.array(x_data_test)


    print("End of loading data..")

    print("Train set size (after balancing) : ", final_df_train_size)
    print("Test set size (after balancing) : ", final_df_test_size)

    #######################
    # 2. Getting model
    #######################

    if not os.path.exists(cfg.backup_model_folder):
        os.makedirs(cfg.backup_model_folder)

    filepath = os.path.join(cfg.backup_model_folder, "{0}-{epoch:02d}.hdf5".format(p_output))
    checkpoint = ModelCheckpoint(filepath, monitor='val_auc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model = models.get_model(n_channels, input_shape, p_tl)
    model.summary()
 
    model.fit(x_data_train, y_dataset_train.values, validation_split=p_val_size, epochs=p_epochs, batch_size=p_batch_size, callbacks=callbacks_list)

    score = model.evaluate(x_data_test, y_dataset_test, batch_size=p_batch_size)

    print("Accuracy score on test dataset ", score)

    if not os.path.exists(cfg.saved_models_folder):
        os.makedirs(cfg.saved_models_folder)

    # save the model into HDF5 file
    model_output_path = os.path.join(cfg.saved_models_folder, p_output + '.json')
    json_model_content = model.to_json()

    with open(model_output_path, 'w') as f:
        print("Model saved into ", model_output_path)
        json.dump(json_model_content, f, indent=4)

    model.save_weights(model_output_path.replace('.json', '.h5'))

    # Get results obtained from model
    y_train_prediction = model.predict(x_data_train)
    y_test_prediction = model.predict(x_data_test)

    y_train_prediction = [1 if x > 0.5 else 0 for x in y_train_prediction]
    y_test_prediction = [1 if x > 0.5 else 0 for x in y_test_prediction]

    acc_train_score = accuracy_score(y_dataset_train, y_train_prediction)
    acc_test_score = accuracy_score(y_dataset_test, y_test_prediction)

    f1_train_score = f1_score(y_dataset_train, y_train_prediction)
    f1_test_score = f1_score(y_dataset_test, y_test_prediction)

    recall_train_score = recall_score(y_dataset_train, y_train_prediction)
    recall_test_score = recall_score(y_dataset_test, y_test_prediction)

    pres_train_score = precision_score(y_dataset_train, y_train_prediction)
    pres_test_score = precision_score(y_dataset_test, y_test_prediction)

    roc_train_score = roc_auc_score(y_dataset_train, y_train_prediction)
    roc_test_score = roc_auc_score(y_dataset_test, y_test_prediction)

    # save model performance
    if not os.path.exists(cfg.results_information_folder):
        os.makedirs(cfg.results_information_folder)

    perf_file_path = os.path.join(cfg.results_information_folder, cfg.csv_model_comparisons_filename)

    with open(perf_file_path, 'a') as f:
        line = p_output + ';' + str(len(dataset_train)) + ';' + str(len(dataset_test)) + ';' \
                        + str(final_df_train_size) + ';' + str(final_df_test_size) + ';' \
                        + str(acc_train_score) + ';' + str(acc_test_score) + ';' \
                        + str(f1_train_score) + ';' + str(f1_test_score) + ';' \
                        + str(recall_train_score) + ';' + str(recall_test_score) + ';' \
                        + str(pres_train_score) + ';' + str(pres_test_score) + ';' \
                        + str(roc_train_score) + ';' + str(roc_test_score) + '\n'
        f.write(line)

if __name__== "__main__":
    main()
