'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
```
data/
    train/
        final/
            final001.png
            final002.png
            ...
        noisy/
            noisy001.png
            noisy002.png
            ...
    validation/
        final/
            final001.png
            final002.png
            ...
        noisy/
            noisy001.png
            noisy002.png
            ...
```
'''
import sys, os, getopt
import json

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
from keras.utils import plot_model

import tensorflow as tf
import numpy as np

from ipfml import tf_model_helper
from ipfml import metrics

# local functions import
import preprocessing_functions

##########################################
# Global parameters (with default value) #
##########################################
img_width, img_height = 100, 100

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 7200
nb_validation_samples = 3600
epochs = 50
batch_size = 16

input_shape = (3, img_width, img_height)

###########################################

'''
Method which returns model to train
@return : DirectoryIterator
'''
def generate_model():

    model = Sequential()

    model.add(Conv2D(60, (2, 1), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Conv2D(40, (2, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Conv2D(30, (2, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Flatten())
    model.add(Dense(150, kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(120, kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(80, kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(40, kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(20, kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model

'''
Method which loads train data
@return : DirectoryIterator
'''
def load_train_data():

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        #rescale=1. / 255,
        #shear_range=0.2,
        #zoom_range=0.2,
        #horizontal_flip=True,
        preprocessing_function=preprocessing_functions.get_s_model_data)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    return train_generator

'''
Method which loads validation data
@return : DirectoryIterator
'''
def load_validation_data():

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(
        #rescale=1. / 255,
        preprocessing_function=preprocessing_functions.get_s_model_data)

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    return validation_generator

def main():

    # update global variable and not local
    global batch_size
    global epochs   
    global img_width
    global img_height
    global input_shape
    global train_data_dir
    global validation_data_dir
    global nb_train_samples
    global nb_validation_samples 

    if len(sys.argv) <= 1:
        print('Run with default parameters...')
        print('classification_cnn_keras_svd.py --directory xxxx --output xxxxx --batch_size xx --epochs xx --img xx')
        sys.exit(2)
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ho:d:b:e:i", ["help", "output=", "directory=", "batch_size=", "epochs=", "img="])
    except getopt.GetoptError:
        # print help information and exit:
        print('classification_cnn_keras_svd.py --directory xxxx --output xxxxx --batch_size xx --epochs xx --img xx')
        sys.exit(2)
    for o, a in opts:
        if o == "-h":
            print('classification_cnn_keras_svd.py --directory xxxx --output xxxxx --batch_size xx --epochs xx --img xx')
            sys.exit()
        elif o in ("-o", "--output"):
            filename = a
        elif o in ("-b", "--batch_size"):
            batch_size = int(a)
        elif o in ("-e", "--epochs"):
            epochs = int(a)
        elif o in ("-d", "--directory"):
            directory = a
        elif o in ("-i", "--img"):
            img_height = int(a)
            img_width = int(a)
        else:
            assert False, "unhandled option"

    # 3 because we have 3 color canals
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    # configuration
    with open('config.json') as json_data:
        d = json.load(json_data)
        train_data_dir = d['train_data_dir']
        validation_data_dir = d['train_validation_dir']

        try:
            nb_train_samples = d[str(img_width)]['nb_train_samples']
            nb_validation_samples = d[str(img_width)]['nb_validation_samples']
        except:
             print("--img parameter missing of invalid (--image_width xx --img_height xx)")
             sys.exit(2)


    # load of model
    model = generate_model()
    model.summary()

    if(directory):
        print('Your model information will be saved into %s...' % directory)

    history = model.fit_generator(
        load_train_data(),
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=load_validation_data(),
        validation_steps=nb_validation_samples // batch_size)

    # if user needs output files
    if(filename):

        # update filename by folder
        if(directory):
            # create folder if necessary
            if not os.path.exists(directory):
                os.makedirs(directory)
            filename = directory + "/" + filename

        # save plot file history
        tf_model_helper.save(history, filename)

        plot_model(model, to_file=str(('%s.png' % filename)), show_shapes=True)
        model.save_weights(str('%s.h5' % filename))


if __name__ == "__main__":
    main()
