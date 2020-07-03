# main imports
import numpy as np
import pandas as pd
import sys, os, argparse
import json

# model imports
import cnn_models as models
import tensorflow as tf
import keras
from keras.models import load_model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from keras.utils import to_categorical

# image processing imports
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg

# counter param
n_counter = 0

def write_progress(progress):
    '''
    Display progress information as progress bar
    '''
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

    parser = argparse.ArgumentParser(description="Train Keras model and save it into .json file")

    parser.add_argument('--data', type=str, help='dataset filename prefix (without .train and .test)', required=True)
    parser.add_argument('--output', type=str, help='output file name desired for model (without .json extension)', required=True)
    parser.add_argument('--tl', type=int, help='use or not of transfer learning (`VGG network`)', default=0, choices=[0, 1])
    parser.add_argument('--batch_size', type=int, help='batch size used as model input', default=64)
    parser.add_argument('--epochs', type=int, help='number of epochs used for training model', default=30)
    parser.add_argument('--chanels', type=int, help="given number of chanels if necessary", default=0)
    parser.add_argument('--size', type=str, help="Size of input images", default="100, 100")
    parser.add_argument('--val_size', type=float, help='percent of validation data during training process', default=0.3)


    args = parser.parse_args()

    p_data_file   = args.data
    p_output      = args.output
    p_tl          = args.tl
    p_batch_size  = args.batch_size
    p_epochs      = args.epochs
    p_chanels     = args.chanels
    p_size        = args.size.split(',')
    p_val_size    = args.val_size

    #p_val_size    = args.val_size
    initial_epoch = 0

    ########################
    # 1. Get and prepare data
    ########################
    print('-----------------------------')
    print("----- Preparing data... -----")
    dataset_train = pd.read_csv(p_data_file + '.train', header=None, sep=";")
    dataset_test = pd.read_csv(p_data_file + '.test', header=None, sep=";")

    print("-- Train set size : ", len(dataset_train))
    print("-- Test set size : ", len(dataset_test))

    # default first shuffle of data
    dataset_train = shuffle(dataset_train)
    dataset_test = shuffle(dataset_test)

    print('-----------------------------')
    print("--Reading all images data...")

    # getting number of chanel
    if p_chanels == 0:
        n_chanels = len(dataset_train[1][1].split('::'))
    else:
        n_chanels = p_chanels

    print("-- Number of chanels : ", n_chanels)
    img_width, img_height = [ int(s) for s in p_size ]

    # specify the number of dimensions
    if K.image_data_format() == 'chanels_first':
        if n_chanels > 1:
            input_shape = (1, n_chanels, img_width, img_height)
        else:
            input_shape = (n_chanels, img_width, img_height)

    else:
        if n_chanels > 1:
            input_shape = (1, img_width, img_height, n_chanels)
        else:
            input_shape = (img_width, img_height, n_chanels)

    # getting weighted class over the whole dataset
    noisy_df_train = dataset_train[dataset_train.iloc[:, 0] == 1]
    not_noisy_df_train = dataset_train[dataset_train.iloc[:, 0] == 0]
    nb_noisy_train = len(noisy_df_train.index)
    nb_not_noisy_train = len(not_noisy_df_train.index)

    noisy_df_test = dataset_test[dataset_test.iloc[:, 0] == 1]
    not_noisy_df_test = dataset_test[dataset_test.iloc[:, 0] == 0]
    nb_noisy_test = len(noisy_df_test.index)
    nb_not_noisy_test = len(not_noisy_df_test.index)

    noisy_samples = nb_noisy_test + nb_noisy_train
    not_noisy_samples = nb_not_noisy_test + nb_not_noisy_train

    total_samples = noisy_samples + not_noisy_samples

    print('-----------------------------')
    print('---- Dataset information ----')
    print('-- noisy:', noisy_samples)
    print('-- not_noisy:', not_noisy_samples)
    print('-- total:', total_samples)
    print('-----------------------------')

    class_weight = {
        0: (noisy_samples / float(total_samples)),
        1: (not_noisy_samples / float(total_samples)),
    }

    final_df_train = dataset_train
    final_df_test = dataset_test
    
    def load_multiple_greyscale(x):
        # update progress
        global n_counter
        n_counter += 1
        write_progress(n_counter / float(total_samples))
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in x.split('::')]

    def load_greyscale(x):
        # update progress
        global n_counter
        n_counter += 1
        write_progress(n_counter / float(total_samples))
        return cv2.imread(x, cv2.IMREAD_GRAYSCALE)

    def load_rgb(x):
        # update progress
        global n_counter
        n_counter += 1
        write_progress(n_counter / float(total_samples))
        return cv2.imread(x)


    print('---- Loading dataset.... ----')
    print('-----------------------------\n')

    # check if specific number of chanels is used
    if p_chanels == 0:
        # `::` is the separator used for getting each img path
        if n_chanels > 1:
            final_df_train[1] = final_df_train[1].apply(lambda x: load_multiple_greyscale(x))
            final_df_test[1] = final_df_test[1].apply(lambda x: load_multiple_greyscale(x))
        else:
            final_df_train[1] = final_df_train[1].apply(lambda x: load_greyscale(x))
            final_df_test[1] = final_df_test[1].apply(lambda x: load_greyscale(x))
    else:
        final_df_train[1] = final_df_train[1].apply(lambda x: load_rgb(x))
        final_df_test[1] = final_df_test[1].apply(lambda x: load_rgb(x))

    # reshape array data
    final_df_train[1] = final_df_train[1].apply(lambda x: np.array(x).reshape(input_shape))
    final_df_test[1] = final_df_test[1].apply(lambda x: np.array(x).reshape(input_shape))

    # shuffle data another time
    final_df_train = shuffle(final_df_train)
    final_df_test = shuffle(final_df_test)

    print('\n-----------------------------')
    print("Validation split is now set at", p_val_size)
    print('-----------------------------')

    # use of the whole data set for training
    x_dataset_train = final_df_train.iloc[:,1:]
    x_dataset_test = final_df_test.iloc[:,1:]

    y_dataset_train = final_df_train.iloc[:,0]
    y_dataset_test = final_df_test.iloc[:,0]

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


    #######################
    # 2. Getting model
    #######################

    # create backup folder for current model
    model_backup_folder = os.path.join(cfg.backup_model_folder, p_output)
    if not os.path.exists(model_backup_folder):
        os.makedirs(model_backup_folder)

    # add of callback models
    filepath = os.path.join(cfg.backup_model_folder, p_output, p_output + "-_{epoch:03d}.h5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, mode='max')
    callbacks_list = [checkpoint]

    
    # check if backup already exists
    backups = sorted(os.listdir(model_backup_folder))

    if len(backups) > 0:
        last_backup_file = backups[-1]
        last_backup_file_path = os.path.join(model_backup_folder, last_backup_file)
        model = load_model(last_backup_file_path)

        # get initial epoch
        initial_epoch = int(last_backup_file.split('_')[-1].replace('.h5', ''))
        print('-----------------------------')  
        print('-- Restore model from backup...')
        print('-- Restart training @epoch:', initial_epoch)
        print('-----------------------------')
    else:
        model = models.get_model(n_chanels, input_shape, p_tl)
        
    model.summary()

    # prepare train and validation dataset
    X_train, X_val, y_train, y_val = train_test_split(x_data_train, y_dataset_train, test_size=p_val_size, shuffle=False)

    y_train_cat = to_categorical(y_train)
    y_val_cat = to_categorical(y_val)
    y_test_cat = to_categorical(y_dataset_test)

    print('-----------------------------')
    print("-- Fitting model with custom class_weight", class_weight)
    print('-----------------------------')
    model.fit(X_train, y_train_cat, 
        validation_data=(X_val, y_val_cat), 
        initial_epoch=initial_epoch, 
        epochs=p_epochs, 
        batch_size=p_batch_size, 
        callbacks=callbacks_list, 
        class_weight=class_weight)

    score = model.evaluate(X_val, y_val_cat, batch_size=p_batch_size)

    print("Accuracy score on val dataset ", score)

    if not os.path.exists(cfg.output_models):
        os.makedirs(cfg.output_models)

    # save the model into H5 file
    model_output_path = os.path.join(cfg.output_models, p_output + '.h5')
    model.save(model_output_path)

    print('Begin of prediction score on the whole dataset:')
    # Get results obtained from model
    y_train_prediction = model.predict(X_train, verbose=1)
    y_val_prediction = model.predict(X_val, verbose=1)
    y_test_prediction = model.predict(x_data_test, verbose=1)

    y_train_prediction = np.argmax(y_train_prediction, axis=1)
    y_val_prediction = np.argmax(y_val_prediction, axis=1)
    y_test_prediction = np.argmax(y_test_prediction, axis=1)

    acc_train_score = accuracy_score(y_train, y_train_prediction)
    acc_val_score = accuracy_score(y_val, y_val_prediction)
    acc_test_score = accuracy_score(y_dataset_test, y_test_prediction)

    roc_train_score = roc_auc_score(y_train, y_train_prediction)
    roc_val_score = roc_auc_score(y_val, y_val_prediction)
    roc_test_score = roc_auc_score(y_dataset_test, y_val_prediction)

    # save model performance
    if not os.path.exists(cfg.output_results_folder):
        os.makedirs(cfg.output_results_folder)

    perf_file_path = os.path.join(cfg.output_results_folder, cfg.csv_model_comparisons_filename)

    # write header if necessary
    if not os.path.exists(perf_file_path):
        with open(perf_file_path, 'w') as f:
            f.write('name;train_acc;val_acc;test_acc;train_auc;val_auc;test_auc;\n')
            
    # add information into file
    with open(perf_file_path, 'a') as f:
        line = p_output + ';' + str(acc_train_score) + ';' + str(acc_val_score) + ';' \
                        + str(acc_test_score) + ';' + str(roc_train_score) + ';' \
                        + str(roc_val_score) + ';' + str(roc_test_score) + '\n'
        f.write(line)

    print("You can now run your model with your own `test` dataset")

if __name__== "__main__":
    main()
