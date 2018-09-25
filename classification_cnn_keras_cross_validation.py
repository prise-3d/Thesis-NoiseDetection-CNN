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
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from sklearn.cross_validation import StratifiedKFold
from keras.utils import plot_model


# dimensions of our images.
img_width, img_height = 100, 100

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 7200
nb_validation_samples = 3600
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)



def create_model():
    # create your model using this function
    model = Sequential()
    model.add(Conv2D(60, (2, 2), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(40, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(20, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(40, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(20, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.05))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.summary()
    plot_model(model, to_file='noise_classification_img100.png', show_shapes=True)
    return model

def load_data():
    # load your data using this function
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    return train_generator

    #validation_generator = test_datagen.flow_from_directory(
    #    validation_data_dir,
    #    target_size=(img_width, img_height),
    #    batch_size=batch_size,
    #    class_mode='binary')

def train_and_evaluate_model(model, data_train, data_test):

    model.fit_generator(
        data_train,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        shuffle=True,
        validation_data=data_test,
        validation_steps=nb_validation_samples // batch_size)

if __name__ == "__main__":
    n_folds = 10

    data_generator = ImageDataGenerator(rescale=1./255, validation_split=0.33)

    # check if possible to not do this thing each time
    train_generator = data_generator.flow_from_directory(train_data_dir, target_size=(img_width, img_height), shuffle=True, seed=13,
                                                         class_mode='binary', batch_size=batch_size, subset="training")

    validation_generator = data_generator.flow_from_directory(train_data_dir, target_size=(img_width, img_height), shuffle=True, seed=13,
                                                         class_mode='binary', batch_size=batch_size, subset="validation")

    model = create_model()
    train_and_evaluate_model(model, train_generator, validation_generator)

    model.save_weights('noise_classification_img100.h5')
