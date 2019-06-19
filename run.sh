#!/bin/bash

metric="svd_reconstruction"

for begin in {80,85,90,95,100,105,110}; do
  for end in {150,160,170,180,190,200}; do

    # python generate_reconstructed_data.py --metric ${metric} --interval "${begin}, ${end}"

    for zone in {6,8,10,12}; do
      OUTPUT_DATA_FILE="${metric}_nb_zones_${zone}_B${begin}_E${end}"

      python generate_dataset.py --output data/${OUTPUT_DATA_FILE} --metric ${metric} --renderer "maxwell" --scenes "A, D, G, H" --interval "${begin}, ${end}" --nb_zones ${zone} --random 1
      
      python train_model.py --data data/${OUTPUT_DATA_FILE} --output ${OUTPUT_DATA_FILE}
    done
  done
done
