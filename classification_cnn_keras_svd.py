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

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
from numpy.linalg import svd
import tensorflow as tf
import numpy as np
from PIL import Image

from scipy import misc
import matplotlib.pyplot as plt
import keras as k

# dimensions of our images.
img_width, img_height = int(100), 1

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 7200
nb_validation_samples = 3600
epochs = 200
batch_size = 30

# configuration
config = tf.ConfigProto(intra_op_parallelism_threads=6, inter_op_parallelism_threads=6, \
                        allow_soft_placement=True, device_count = {'CPU': 6})
session = tf.Session(config=config)
K.set_session(session)

def svd_singular(image):
    U, s, V = svd(image, full_matrices=False)
    s = s[0:img_width]
    result = s.reshape([img_width, 1, 1]) # one shape per canal
    return result

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()

model.add(Conv2D(100, (2, 1), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 1)))

model.add(Conv2D(80, (2, 1)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 1)))

model.add(Conv2D(50, (2, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 1)))

model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(300, kernel_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(30, kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(100, kernel_regularizer=l2(0.01)))
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

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    #rescale=1. / 255,
    #shear_range=0.2,
    #zoom_range=0.2,
    #horizontal_flip=True,
    preprocessing_function=svd_singular)
# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(
    #rescale=1. / 255,
    preprocessing_function=svd_singular)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


model.summary()
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('noise_classification_img100.h5')
