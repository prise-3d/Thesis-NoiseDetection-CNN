#!/bin/bash

size=$1

if [ -z ${size} ]; then
  echo "Run algorithms with image of size ${size}.."
else 
  echo "Need size parameter : ./run.sh 20"; 
fi

python classification_cnn_keras_svd.py --directory ../models/$size/ --output svd_model --batch_size 32 --epochs 150 --img $size
python classification_cnn_keras.py --directory ../models/$size/ --output cnn_model --batch_size 32 --epochs 150 --img $size
python classification_cnn_keras_cross_validation.py --directory ../models/$size/ --output cnn_cross_validation_model --batch_size 32 --epochs 150 --img $size