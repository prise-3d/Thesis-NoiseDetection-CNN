import numpy as np
import pandas as pd
import sys, os, argparse
import json

import cv2

from sklearn.utils import shuffle

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import backend as K
from keras.utils import plot_model

from modules.utils import config as cfg
from sklearn.metrics import roc_auc_score

img_width, img_height = 200, 200
batch_size = 32

# 1 because we have 1 color canal
if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)


def generate_model(_input_shape):

    model = Sequential()

    model.add(Conv2D(60, (2, 2), input_shape=_input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(40, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(20, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(140))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(120))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(80))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(40))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(20))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model


def main():

    parser = argparse.ArgumentParser(description="Train Keras model and save it into .json file")

    parser.add_argument('--data', type=str, help='dataset filename prefix (without .train and .test)')
    parser.add_argument('--output', type=str, help='output file name desired for model (without .json extension)')

    args = parser.parse_args()

    p_data_file = args.data
    p_output    = args.output

    ########################
    # 1. Get and prepare data
    ########################
    print("Preparing data...")
    dataset_train = pd.read_csv(p_data_file + '.train', header=None, sep=";")
    dataset_test = pd.read_csv(p_data_file + '.test', header=None, sep=";")

    # default first shuffle of data
    dataset_train = shuffle(dataset_train)
    dataset_test = shuffle(dataset_test)

    print("Reading all images data...")
    dataset_train[1] = dataset_train[1].apply(lambda x: cv2.imread(x, cv2.IMREAD_GRAYSCALE).reshape(input_shape))
    dataset_test[1] = dataset_test[1].apply(lambda x: cv2.imread(x, cv2.IMREAD_GRAYSCALE).reshape(input_shape))
    

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
    print("End of loading data..")

    print(x_data_train.shape)
    print(x_data_train[0])

    #######################
    # 2. Getting model
    #######################

    model = generate_model(input_shape)
    model.summary()

    model.fit(x_data_train, y_dataset_train.values, validation_split=0.20, epochs=cfg.keras_epochs, batch_size=cfg.keras_batch)

    score = model.evaluate(x_dataset_test, y_dataset_test, batch_size=cfg.keras_batch)

    if not os.path.exists(cfg.saved_models_folder):
        os.makedirs(cfg.saved_models_folder)

    # save the model into HDF5 file
    model_output_path = os.path.join(cfg.saved_models_folder, p_output + '.json')
    json_model_content = model.to_json()

    with open(model_output_path, 'w') as f:
        print("Model saved into ", model_output_path)
        json.dump(json_model_content, f, indent=4)

    model.save_weights(model_output_path.replace('.json', '.h5'))

    # Save results obtained from model
    y_test_prediction = model.predict(x_dataset_test)
    print("Metrics : ", model.metrics_names)
    print("Prediction : ", score)
    print("ROC AUC : ", roc_auc_score(y_dataset_test, y_test_prediction))

if __name__== "__main__":
    main()
