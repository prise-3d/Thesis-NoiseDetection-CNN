#!/bin/bash

erased=$1

# file which contains model names we want to use for simulation
file_path="results/models_comparisons.csv"

if [ "${erased}" == "Y" ]; then
    echo "Previous data file erased..."
    rm ${file_path}
    mkdir -p results
    touch ${file_path}

    # add of header
    echo 'model_name; global_train_size; global_test_size; filtered_train_size; filtered_test_size; f1_train; f1_test; recall_train; recall_test; presicion_train; precision_test; acc_train; acc_test; roc_auc_train; roc_auc_test;' >> ${file_path}
fi

renderer="all"
scenes="A, B, C, D, E, F, G, H, I"

svd_metric="svd_reconstruction"
ipca_metric="ipca_reconstruction"
fast_ica_metric="fast_ica_reconstruction"

all_features="${svd_metric},${ipca_metric},${fast_ica_metric}"


# RUN LATER
# compute using all transformation methods
ipca_batch_size=55
begin=100
end=200
ipca_component=30
fast_ica_component=60
zone=12


OUTPUT_DATA_FILE="${svd_metric}_B${begin}_E${end}_${ipca_metric}__N${ipca_component}_BS${ipca_batch_size}_${fast_ica_metric}_N${fast_ica_component}_nb_zones_${zone}"

python generate/generate_reconstructed_data.py --features ${svd_metric} --params "${begin}, ${end}"

python generate/generate_reconstructed_data.py --features ${ipca_component} --params "${component},${ipca_batch_size}"

python generate/generate_reconstructed_data.py --features ${fast_ica_component} --params "${component}"


if grep -xq "${OUTPUT_DATA_FILE}" "${file_path}"; then
  
  echo "Transformation combination model ${OUTPUT_DATA_FILE} already generated"

else

  echo "Run computation for Transformation combination model ${OUTPUT_DATA_FILE}"

  params="${begin}, ${end} :: ${ipca_component}, ${ipca_batch_size} :: ${fast_ica_component}"

  python generate/generate_dataset.py --output data/${OUTPUT_DATA_FILE} --metric ${all_features} --renderer ${renderer} --scenes ${scenes} --params "${params}" --nb_zones ${zone} --random 1
  
  python train_model.py --data data/${OUTPUT_DATA_FILE} --output ${OUTPUT_DATA_FILE} --tl 1 &
fi
