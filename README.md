# Noise detection project

## Requirements

```
pip install -r requirements.txt
```

## How to use

Generate dataset (run only once time or clean data folder before) :
```
python generate_dataset.py
```

It will split scenes and generate all data you need for your neural network.
You can specify the number of sub images you want in the script by modifying **_NUMBER_SUB_IMAGES_** variables.

There are 3 kinds of Neural Networks :
- **classification_cnn_keras.py** : *based on cropped images and do convolution*
- **classification_cnn_keras_cross_validation.py** : *based on cropped images and do convolution. Data are randomly split for training*
- **classification_cnn_keras_svd.py** : *based on svd metrics of image*

Note that the image input size need to change in you used specific size for your croped images.

After your built your neural network in classification_cnn_keras.py, you just have to run it :
```
classification_cnn_keras_svd.py --directory xxxx --output xxxxx --batch_size xx --epochs xx --img xx (or --image_width xx --img_height xx)
```

A config file in json is available and keeps in memory all image sizes available.

## Modules

This project contains modules :
- **modules/image_metrics** : *where all computed metrics function are developed*
- **modules/model_helper** : *contains helpful function to save or display model information and performance*

All these modules will be enhanced during development of the project

## How to contribute

This git project uses [git-flow](https://danielkummer.github.io/git-flow-cheatsheet/) implementation. You are free to contribute to it.
