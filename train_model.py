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

    parser = argparse.ArgumentParser(description="Train Keras model and save it into .json file")

    parser.add_argument('--data', type=str, help='dataset filename prefix (without .train and .val)', required=True)
    parser.add_argument('--output', type=str, help='output file name desired for model (without .json extension)', required=True)
    parser.add_argument('--tl', type=int, help='use or not of transfer learning (`VGG network`)', default=0, choices=[0, 1])
    parser.add_argument('--batch_size', type=int, help='batch size used as model input', default=cfg.keras_batch)
    parser.add_argument('--epochs', type=int, help='number of epochs used for training model', default=cfg.keras_epochs)
    #parser.add_argument('--val_size', type=float, help='percent of validation data during training process', default=cfg.val_dataset_size)


    args = parser.parse_args()

    p_data_file   = args.data
    p_output      = args.output
    p_tl          = args.tl
    p_batch_size  = args.batch_size
    p_epochs      = args.epochs
    #p_val_size    = args.val_size
    initial_epoch = 0
        
    ########################
    # 1. Get and prepare data
    ########################
    print("Preparing data...")
    dataset_train = pd.read_csv(p_data_file + '.train', header=None, sep=";")
    dataset_val = pd.read_csv(p_data_file + '.val', header=None, sep=";")

    print("Train set size : ", len(dataset_train))
    print("val set size : ", len(dataset_val))

    # default first shuffle of data
    dataset_train = shuffle(dataset_train)
    dataset_val = shuffle(dataset_val)

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
        dataset_val[1] = dataset_val[1].apply(lambda x: [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in x.split('::')])
    else:
        dataset_train[1] = dataset_train[1].apply(lambda x: cv2.imread(x, cv2.IMREAD_GRAYSCALE))
        dataset_val[1] = dataset_val[1].apply(lambda x: cv2.imread(x, cv2.IMREAD_GRAYSCALE))

    # reshape array data
    dataset_train[1] = dataset_train[1].apply(lambda x: np.array(x).reshape(input_shape))
    dataset_val[1] = dataset_val[1].apply(lambda x: np.array(x).reshape(input_shape))

    # get dataset with equal number of classes occurences
    noisy_df_train = dataset_train[dataset_train.ix[:, 0] == 1]
    not_noisy_df_train = dataset_train[dataset_train.ix[:, 0] == 0]
    nb_noisy_train = len(noisy_df_train.index)

    noisy_df_val = dataset_val[dataset_val.ix[:, 0] == 1]
    not_noisy_df_val = dataset_val[dataset_val.ix[:, 0] == 0]
    nb_noisy_val = len(noisy_df_val.index)

    final_df_train = pd.concat([not_noisy_df_train[0:nb_noisy_train], noisy_df_train])
    final_df_val = pd.concat([not_noisy_df_val[0:nb_noisy_val], noisy_df_val])

    # shuffle data another time
    final_df_train = shuffle(final_df_train)
    final_df_val = shuffle(final_df_val)

    final_df_train_size = len(final_df_train.index)
    final_df_val_size = len(final_df_val.index)

    validation_split = final_df_val_size / (final_df_train_size + final_df_val_size)
    print("Validation size is based of `.val` content")
    print("Validation split is now set at", )

    # use of the whole data set for training
    x_dataset_train = final_df_train.ix[:,1:]
    x_dataset_val = final_df_val.ix[:,1:]

    y_dataset_train = final_df_train.ix[:,0]
    y_dataset_val = final_df_val.ix[:,0]

    x_data_train = []
    for item in x_dataset_train.values:
        #print("Item is here", item)
        x_data_train.append(item[0])

    x_data_train = np.array(x_data_train)

    x_data_val = []
    for item in x_dataset_val.values:
        #print("Item is here", item)
        x_data_val.append(item[0])

    x_data_val = np.array(x_data_val)

    print("End of loading data..")

    print("Train set size (after balancing) : ", final_df_train_size)
    print("val set size (after balancing) : ", final_df_val_size)

    #######################
    # 2. Getting model
    #######################

    # create backup folder for current model
    model_backup_folder = os.path.join(cfg.backup_model_folder, p_output)
    if not os.path.exists(model_backup_folder):
        os.makedirs(model_backup_folder)

    # add of callback models
    filepath = os.path.join(cfg.backup_model_folder, p_output, p_output + "__{epoch:02d}.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_auc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model = models.get_model(n_channels, input_shape, p_tl)
    model.summary()

    # check if backup already exists
    backups = sorted(os.listdir(model_backup_folder))

    if len(backups) > 0:
        # TODO : check of initial epoch
        last_backup = backups[-1]
        last_epoch = int(last_backup.split('__')[1].replace('.hdf5', ''))
        initial_epoch = last_epoch
        print("Previous backup model found.. ")
        print("Restart from epoch ", last_epoch)

    # concatenate train and validation data (`validation_split` param will do the separation into keras model)
    y_data = y_dataset_train.values + y_dataset_val.values
    x_data = x_data_train + y_data_train

    # validation split parameter will use the last `%` data, so here, data will really validate our model
    model.fit(x_data_train, y_dataset_train.values, validation_split=validation_split, initial_epoch=initial_epoch, epochs=p_epochs, batch_size=p_batch_size, callbacks=callbacks_list)

    score = model.evaluate(x_data_val, y_dataset_val, batch_size=p_batch_size)

    print("Accuracy score on val dataset ", score)

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
    y_val_prediction = model.predict(x_data_val)

    y_train_prediction = [1 if x > 0.5 else 0 for x in y_train_prediction]
    y_val_prediction = [1 if x > 0.5 else 0 for x in y_val_prediction]

    acc_train_score = accuracy_score(y_dataset_train, y_train_prediction)
    acc_val_score = accuracy_score(y_dataset_val, y_val_prediction)

    f1_train_score = f1_score(y_dataset_train, y_train_prediction)
    f1_val_score = f1_score(y_dataset_val, y_val_prediction)

    recall_train_score = recall_score(y_dataset_train, y_train_prediction)
    recall_val_score = recall_score(y_dataset_val, y_val_prediction)

    pres_train_score = precision_score(y_dataset_train, y_train_prediction)
    pres_val_score = precision_score(y_dataset_val, y_val_prediction)

    roc_train_score = roc_auc_score(y_dataset_train, y_train_prediction)
    roc_val_score = roc_auc_score(y_dataset_val, y_val_prediction)

    # save model performance
    if not os.path.exists(cfg.results_information_folder):
        os.makedirs(cfg.results_information_folder)

    perf_file_path = os.path.join(cfg.results_information_folder, cfg.csv_model_comparisons_filename)

    # write header if necessary
    if not os.path.exists(perf_file_path):
        with open(perf_file_path, 'w') as f:
            f.write(cfg.perf_train_header_file)
            
    # add information into file
    with open(perf_file_path, 'a') as f:
        line = p_output + ';' + str(len(dataset_train)) + ';' + str(len(dataset_val)) + ';' \
                        + str(final_df_train_size) + ';' + str(final_df_val_size) + ';' \
                        + str(acc_train_score) + ';' + str(acc_val_score) + ';' \
                        + str(f1_train_score) + ';' + str(f1_val_score) + ';' \
                        + str(recall_train_score) + ';' + str(recall_val_score) + ';' \
                        + str(pres_train_score) + ';' + str(pres_val_score) + ';' \
                        + str(roc_train_score) + ';' + str(roc_val_score) + '\n'
        f.write(line)

    print("You can now run your model with your own `test` dataset")

if __name__== "__main__":
    main()
