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

# First compute svd_reconstruction

for begin in {80,85,90,95,100,105,110}; do
  for end in {150,160,170,180,190,200}; do

    python generate_reconstructed_data.py --metric ${svd_metric} --param "${begin}, ${end}"

    for zone in {6,8,10,12}; do
      OUTPUT_DATA_FILE="${svd_metric}_nb_zones_${zone}_B${begin}_E${end}"

      if grep -xq "${OUTPUT_DATA_FILE}" "${file_path}"; then
        
        echo "SVD model ${OUTPUT_DATA_FILE} already generated"
      
      else
      
        echo "Run computation for SVD model ${OUTPUT_DATA_FILE}"

        python generate_dataset.py --output data/${OUTPUT_DATA_FILE} --metric ${svd_metric} --renderer ${renderer} --scenes ${scenes} --param "${begin}, ${end}" --nb_zones ${zone} --random 1
        
        python train_model_2D.py --data data/${OUTPUT_DATA_FILE} --output ${OUTPUT_DATA_FILE}
      fi
    done
  done
done


# computation of ipca_reconstruction
ipca_batch_size=25

for component in {50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200}; do
  python generate_reconstructed_data.py --metric ${ipca_metric} --param "${component},${ipca_batch_size}"

  for zone in {6,8,10,12}; do
    OUTPUT_DATA_FILE="${ipca_metric}_nb_zones_${zone}_N${component}_BS${ipca_batch_size}"

    if grep -xq "${OUTPUT_DATA_FILE}" "${file_path}"; then
      
      echo "IPCA model ${OUTPUT_DATA_FILE} already generated"
    
    else
    
      echo "Run computation for IPCA model ${OUTPUT_DATA_FILE}"

      python generate_dataset.py --output data/${OUTPUT_DATA_FILE} --metric ${ipca_metric} --renderer ${renderer} --scenes ${scenes} --param "${component},${ipca_batch_size}" --nb_zones ${zone} --random 1
      
      python train_model_2D.py --data data/${OUTPUT_DATA_FILE} --output ${OUTPUT_DATA_FILE}
    fi
  done
done


# computation of fast_ica_reconstruction

for component in {50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200}; do
  python generate_reconstructed_data.py --metric ${fast_ica_metric} --param "${component}"

  for zone in {6,8,10,12}; do
    OUTPUT_DATA_FILE="${fast_ica_metric}_nb_zones_${zone}_N${component}"

    if grep -xq "${OUTPUT_DATA_FILE}" "${file_path}"; then
      
      echo "Fast ICA model ${OUTPUT_DATA_FILE} already generated"
    
    else
    
      echo "Run computation for Fast ICA model ${OUTPUT_DATA_FILE}"

      python generate_dataset.py --output data/${OUTPUT_DATA_FILE} --metric ${fast_ica_metric} --renderer ${renderer} --scenes ${scenes} --param "${component}" --nb_zones ${zone} --random 1
      
      python train_model_2D.py --data data/${OUTPUT_DATA_FILE} --output ${OUTPUT_DATA_FILE}
    fi
  done
done
