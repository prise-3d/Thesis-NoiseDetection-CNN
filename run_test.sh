#!/bin/bash

erased=$1

# file which contains model names we want to use for simulation
file_path="models_info/models_comparisons.csv"

if [ "${erased}" == "Y" ]; then
    echo "Previous data file erased..."
    rm ${file_path}
    mkdir -p models_info
    touch ${file_path}

    # add of header
    echo 'model_name; global_train_size; global_test_size; filtered_train_size; filtered_test_size; f1_train; f1_test; recall_train; recall_test; presicion_train; precision_test; acc_train; acc_test; roc_auc_train; roc_auc_test;' >> ${file_path}
fi

renderer="maxwell"
scenes="A, D, G, H"

svd_metric="svd_reconstruction"
ipca_metric="ipca_reconstruction"
fast_ica_metric="fast_ica_reconstruction"

metrics="${svd_metric},${ipca_metric},${fast_ica_metric}"

python generate_reconstructed_data.py --metric ${svd_metric} --param "100, 200"
python generate_reconstructed_data.py --metric ${ipca_reconstruction} --param "50, 10"
python generate_reconstructed_data.py --metric ${fast_ica_metric} --param "50"

OUTPUT_DATA_FILE="test_3D_model"

python generate_dataset_3D.py --output data/${OUTPUT_DATA_FILE} --metrics ${metrics} --renderer ${renderer} --scenes ${scenes} --params "100, 200 :: 50, 10 :: 50" --nb_zones ${zone} --random 1
python train_model.py --data data/${OUTPUT_DATA_FILE} --output ${OUTPUT_DATA_FILE} --n_channels 3