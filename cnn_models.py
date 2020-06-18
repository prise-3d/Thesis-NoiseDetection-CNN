# main imports
import sys

# model imports
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Conv3D, MaxPooling3D, AveragePooling3D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.applications.vgg19 import VGG19
from keras import backend as K
import tensorflow as tf

# configuration and modules imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
#from models import metrics


def generate_model_2D(_input_shape, _weights_file=None):

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
    model.add(Dropout(0.5))

    # model.add(Dense(120))
    # model.add(Activation('sigmoid'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))

    model.add(Dense(80))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(40))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(20))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(2))
    model.add(Activation('softmax'))

    # reload weights if exists
    if _weights_file is not None:
        model.load_weights(_weights_file)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  #metrics=['accuracy', metrics.auc])
                  metrics=['accuracy'])

    return model


def generate_model_3D(_input_shape, _weights_file=None):

    model = Sequential()

    print(_input_shape)

    model.add(Conv3D(60, (1, 2, 2), input_shape=_input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))

    model.add(Conv3D(40, (1, 2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))

    model.add(Conv3D(20, (1, 2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))

    model.add(Flatten())

    model.add(Dense(140))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(120))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(80))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(40))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(20))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(2))
    model.add(Activation('sigmoid'))

    # reload weights if exists
    if _weights_file is not None:
        model.load_weights(_weights_file)

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  #metrics=['accuracy', metrics.auc])
                  metrics=['accuracy'])

    return model


# using transfer learning (VGG19)
def generate_model_3D_TL(_input_shape, _weights_file=None):

    # load pre-trained model
    model = VGG19(weights='imagenet', include_top=False, input_shape=_input_shape)
    # display model layers
    model.summary()

    # do not train convolutional layers
    for layer in model.layers[:5]:
        layer.trainable = False

    '''predictions_model = Sequential(model)

    predictions_model.add(Flatten(model.output))

    predictions_model.add(Dense(1024))
    predictions_model.add(Activation('relu'))
    predictions_model.add(BatchNormalization())
    predictions_model.add(Dropout(0.5))

    predictions_model.add(Dense(512))
    predictions_model.add(Activation('relu'))
    predictions_model.add(BatchNormalization())
    predictions_model.add(Dropout(0.5))

    predictions_model.add(Dense(256))
    predictions_model.add(Activation('relu'))
    predictions_model.add(BatchNormalization())
    model.add(Dropout(0.5))

    predictions_model.add(Dense(100))
    predictions_model.add(Activation('relu'))
    predictions_model.add(BatchNormalization())
    predictions_model.add(Dropout(0.5))

    predictions_model.add(Dense(20))
    predictions_model.add(Activation('relu'))
    predictions_model.add(BatchNormalization())
    predictions_model.add(Dropout(0.5))

    predictions_model.add(Dense(1))
    predictions_model.add(Activation('sigmoid'))'''

    # adding custom Layers 
    x = model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(16, activation="relu")(x)
    predictions = Dense(1, activation="softmax")(x)

    # creating the final model 
    model_final = Model(input=model.input, output=predictions)

    model_final.summary()

    # reload weights if exists
    if _weights_file is not None:
        model.load_weights(_weights_file)

    model_final.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                #   metrics=['accuracy', metrics.auc])
                  metrics=['accuracy'])

    return model_final


def get_model(n_channels, _input_shape, _tl=False, _weights_file=None):
    
    if _tl:
        if n_channels == 3:
            return generate_model_3D_TL(_input_shape, _weights_file)
        else:
            print("Can't use transfer learning with only 1 channel")

    if n_channels == 1:
        return generate_model_2D(_input_shape, _weights_file)

    if n_channels >= 2:
        return generate_model_3D(_input_shape, _weights_file)