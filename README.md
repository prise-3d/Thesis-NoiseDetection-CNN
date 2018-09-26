# Noise detection project

## Requirements

```
pip install -r requirements.txt
```

## How to use

Generate dataset (run only once time) :
```
python generate_dataset.py
```

It will split scenes and generate all data you need for your neural network.
You can specify the number of sub images you want in the script by modifying NUMBER_SUB_IMAGES variables.


There are 3 kind of Neural Networks :
- classification_cnn_keras.py : based croped on images
- classification_cnn_keras_crossentropy.py : based croped on images which are randomly split for training
- classification_cnn_keras_svd.py : based on svd metrics of image

Note that the image input size need to change in you used specific size for your croped images.

After your built your neural network in classification_cnn_keras.py, you just have to run it :
```
python classification_cnn_keras.py
```

## Modules

This project contains modules :
- modules/image_metrics : where all computed metrics function are developed
- modules/model_helper : contains helpful function to save or display model information and performance

All these modules will grow during developement of the project

## How to contribute

This git project uses [git-flow](https://danielkummer.github.io/git-flow-cheatsheet/) implementation. You are free to contribute to it.
