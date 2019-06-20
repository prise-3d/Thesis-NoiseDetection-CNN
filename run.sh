#!/bin/bash

erased=$1

if [ "${erased}" == "Y" ]; then
    echo "Previous data file erased..."
    rm ${file_path}
    mkdir -p models_info
    touch ${file_path}

    # add of header
    echo 'model_name; global_train_size; global_test_size; filtered_train_size; filtered_test_size; f1_train; f1_test; recall_train; recall_test; presicion_train; precision_test; acc_train; acc_test; roc_auc_train; roc_auc_test;' >> ${file_path}
fi

metric="svd_reconstruction"

# file which contains model names we want to use for simulation
comparisons_models="models_info/models_comparisons.csv"

for begin in {80,85,90,95,100,105,110}; do
  for end in {150,160,170,180,190,200}; do

    # python generate_reconstructed_data.py --metric ${metric} --interval "${begin}, ${end}"

    for zone in {6,8,10,12}; do
      OUTPUT_DATA_FILE="${metric}_nb_zones_${zone}_B${begin}_E${end}"

      if grep -xq "${OUTPUT_DATA_FILE}" "${comparisons_models}"; then
        
        echo "Run simulation for model ${OUTPUT_DATA_FILE}"

        python generate_dataset.py --output data/${OUTPUT_DATA_FILE} --metric ${metric} --renderer "maxwell" --scenes "A, D, G, H" --interval "${begin}, ${end}" --nb_zones ${zone} --random 1
        
        python train_model.py --data data/${OUTPUT_DATA_FILE} --output ${OUTPUT_DATA_FILE}
    done
  done
done
