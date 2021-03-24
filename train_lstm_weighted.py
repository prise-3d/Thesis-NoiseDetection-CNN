# main imports
import argparse, sys
import numpy as np
import pandas as pd
import os
import ctypes
from PIL import Image
import cv2

from keras import backend as K
import matplotlib.pyplot as plt
from ipfml import utils

# dl imports
from keras.layers import Dense, Dropout, LSTM, Embedding, GRU, BatchNormalization, ConvLSTM2D, Conv3D, Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score, accuracy_score
import tensorflow as tf
from keras import backend as K
import sklearn
from sklearn.model_selection import train_test_split
from joblib import dump

import config as cfg

# global variables
n_counter = 0
total_samples = 0

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


def build_input(df, seq_norm, p_chanels):
    """Convert dataframe to numpy array input with timesteps as float array
    
    Arguments:
        df: {pd.Dataframe} -- Dataframe input
        seq_norm: {bool} -- normalize or not seq input data by features
    
    Returns:
        {np.ndarray} -- input LSTM data as numpy array
    """

    global n_counter
    global total_samples
    arr = []

    # for each input line
    for row in df.iterrows():

        seq_arr = []

        # for each sequence data input
        for column in row[1]:

            seq_elems = []

            # for each element in sequence data
            for i, img_path in enumerate(column):

                # seq_elems.append(np.array(img).flatten())
                if p_chanels[i] > 1:
                    img = cv2.imread(img_path)
                else:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (50, 50))
                
                # normalization of images
                seq_elems.append(np.array(img, 'float16') / 255.)

            #seq_arr.append(np.array(seq_elems).flatten())
            seq_arr.append(np.array(seq_elems))
            
        arr.append(seq_arr)

        # update progress
        n_counter += 1
        write_progress(n_counter / float(total_samples))

    arr = np.array(arr)
    print(arr.shape)

    # final_arr = []
    # for v in arr:
    #     v_data = []
    #     for vv in v:
    #         #scaled_vv = np.array(vv, 'float') - np.mean(np.array(vv, 'float'))
    #         #v_data.append(scaled_vv)
    #         v_data.append(vv)
        
    #     final_arr.append(v_data)
    
    final_arr = np.array(arr, 'float16')

    # check if sequence normalization is used
    if seq_norm:

        if final_arr.ndim > 2:
            n, s, f = final_arr.shape
            for index, seq in enumerate(final_arr):
                
                for i in range(f):
                    final_arr[index][:, i] = utils.normalize_arr_with_range(seq[:, i])

            

    return final_arr

def create_model(_input_shape):
    print ('Creating model...')
    model = Sequential()
    
    # model.add(Conv3D(60, (1, 2, 2), input_shape=input_shape))
    # model.add(Activation('relu'))
    # model.add(MaxPooling3D(pool_size=(1, 2, 2)))

    #model.add(Embedding(input_dim = 1000, output_dim = 50, input_length=input_length))
    # model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), input_shape=input_shape, units=256, activation='sigmoid', recurrent_activation='hard_sigmoid'))
    # model.add(Dropout(0.4))
    # model.add(GRU(units=128, activation='sigmoid', recurrent_activation='hard_sigmoid'))
    # model.add(Dropout(0.4))
    # model.add(Dense(1, activation='sigmoid'))

    model.add(ConvLSTM2D(filters=100, kernel_size=(3, 3),
                   input_shape=_input_shape,
                   dropout=0.5,
                   #recurrent_dropout=0.5,
                   padding='same', return_sequences=True))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=30, kernel_size=(3, 3),
                    dropout=0.5,
                    #recurrent_dropout=0.5,
                    padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Conv3D(filters=15, kernel_size=(3, 3, 3),
                activation='sigmoid',
                padding='same', data_format='channels_last'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print ('-- Compiling...')

    return model


def main():

    # get this variable as global
    global total_samples

    parser = argparse.ArgumentParser(description="Read and compute training of LSTM model")

    parser.add_argument('--train', type=str, help='input train dataset', required=True)
    parser.add_argument('--test', type=str, help='input test dataset', required=True)
    parser.add_argument('--output', type=str, help='output model name', required=True)
    parser.add_argument('--chanels', type=str, help="given number of ordered chanels (example: '1,3,3') for each element of window", required=True)
    parser.add_argument('--epochs', type=int, help='number of expected epochs', default=30)
    parser.add_argument('--batch_size', type=int, help='expected batch size for training model', default=64)
    parser.add_argument('--seq_norm', type=int, help='normalization sequence by features', choices=[0, 1], default=0)

    args = parser.parse_args()

    p_train        = args.train
    p_test         = args.test
    p_output       = args.output
    p_chanels     = list(map(int, args.chanels.split(',')))
    p_epochs       = args.epochs
    p_batch_size   = args.batch_size
    p_seq_norm     = bool(args.seq_norm)

    print('-----------------------------')
    print("----- Preparing data... -----")
    dataset_train = pd.read_csv(p_train, header=None, sep=';')
    dataset_test = pd.read_csv(p_test, header=None, sep=';')

    print("-- Train set size : ", len(dataset_train))
    print("-- Test set size : ", len(dataset_test))

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
        0: noisy_samples / float(total_samples),
        1: (not_noisy_samples / float(total_samples)),
    }

    # shuffle data
    final_df_train = sklearn.utils.shuffle(dataset_train)
    final_df_test = sklearn.utils.shuffle(dataset_test)

    print('---- Loading dataset.... ----')
    print('-----------------------------\n')

    # split dataset into X_train, y_train, X_test, y_test
    X_train_all = final_df_train.loc[:, 1:].apply(lambda x: x.astype(str).str.split('::'))
    X_train_all = build_input(X_train_all, p_seq_norm, p_chanels)
    y_train_all = final_df_train.loc[:, 0].astype('int')

    input_shape = (X_train_all.shape[1], X_train_all.shape[2], X_train_all.shape[3], X_train_all.shape[4])
    
    
    print('\n-----------------------------')
    print('-- Training data input shape', input_shape)
    print('-----------------------------')

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
        model = load_model(last_backup_file)

        # get initial epoch
        initial_epoch = int(last_backup_file.split('_')[-1].replace('.h5', ''))
        print('-----------------------------')  
        print('-- Restore model from backup...')
        print('-- Restart training @epoch:', initial_epoch)
        print('-----------------------------')
    else:
        model = create_model(input_shape)
    model.summary()

    # prepare train and validation dataset
    X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.3, shuffle=False)

    print("-- Fitting model with custom class_weight", class_weight)
    print('-----------------------------')
    history = model.fit(X_train, y_train, batch_size=p_batch_size, epochs=p_epochs, validation_data=(X_val, y_val), verbose=1, shuffle=True, class_weight=class_weight)

    # list all data in history
    # print(history.history.keys())
    # # summarize history for accuracy
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

    # train_score, train_acc = model.evaluate(X_train, y_train, batch_size=1)

    # print(train_acc)
    y_train_predict = model.predict(X_train, batch_size=1, verbose=1)
    y_val_predict = model.predict(X_val, batch_size=1, verbose=1)

    y_train_predict = [ 1 if l > 0.5 else 0 for l in y_train_predict ]
    y_val_predict = [ 1 if l > 0.5 else 0 for l in y_val_predict ]

    auc_train = roc_auc_score(y_train, y_train_predict)
    auc_val = roc_auc_score(y_val, y_val_predict)

    acc_train = accuracy_score(y_train, y_train_predict)
    acc_val = accuracy_score(y_val, y_val_predict)

    # remove unused variables
    del X_train
    del y_train
    
    X_test = final_df_test.loc[:, 1:].apply(lambda x: x.astype(str).str.split('::'))
    X_test = build_input(X_test, p_seq_norm, p_chanels)
    y_test = final_df_test.loc[:, 0].astype('int')

    y_test_predict = model.predict(X_test, batch_size=1, verbose=1)
    y_test_predict = [ 1 if l > 0.5 else 0 for l in y_test_predict ]

    acc_test = accuracy_score(y_test, y_test_predict)
    auc_test = roc_auc_score(y_test, y_test_predict)

    print('Train ACC:', acc_train)
    print('Train AUC', auc_train)
    print('Val ACC:', acc_val)
    print('Val AUC', auc_val)
    print('Test ACC:', acc_test)
    print('Test AUC:', auc_test)

    # save acc metric information
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    model_history = os.path.join(cfg.output_results_folder, p_output + '.png')
    plt.savefig(model_history)

    # save model using keras API
    if not os.path.exists(cfg.output_models):
        os.makedirs(cfg.output_models)

    model.save(os.path.join(cfg.output_models, p_output + '.h5'))

    # save model results
    if not os.path.exists(cfg.output_results_folder):
        os.makedirs(cfg.output_results_folder)
    
    results_filename_path = os.path.join(cfg.output_results_folder, cfg.results_filename)

    if not os.path.exists(results_filename_path):
        with open(results_filename_path, 'w') as f:
            f.write('name;train_acc;val_acc;test_acc;train_auc;val_auc;test_auc;\n')

    with open(results_filename_path, 'a') as f:
        f.write(p_output + ';' + str(acc_train) + ';' + str(acc_val) + ';' + str(acc_test) + ';' \
             + str(auc_train) + ';' + str(auc_val) + ';' + str(auc_test) + '\n')

if __name__ == "__main__":
    main()