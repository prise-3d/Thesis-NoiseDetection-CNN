#!/bin/bash

erased=$1

# file which contains model names we want to use for simulation
file_path="results/models_comparisons.csv"

if [ "${erased}" == "Y" ]; then
    echo "Previous data file erased..."
    rm ${file_path}
    mkdir -p models_info
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

# First compute svd_reconstruction

for begin in {80,85,90,95,100,105,110}; do
  for end in {150,160,170,180,190,200}; do
  
    python generate/generate_reconstructed_data.py --features ${svd_metric} --params "${begin}, ${end}"

    for zone in {6,8,10,12}; do
      OUTPUT_DATA_FILE="${svd_metric}_nb_zones_${zone}_B${begin}_E${end}"

      if grep -xq "${OUTPUT_DATA_FILE}" "${file_path}"; then
        
        echo "SVD model ${OUTPUT_DATA_FILE} already generated"
      
      else
      
        echo "Run computation for SVD model ${OUTPUT_DATA_FILE}"

        python generate/generate_dataset.py --output data/${OUTPUT_DATA_FILE} --features ${svd_metric} --renderer ${renderer} --scenes ${scenes} --params "${begin}, ${end}" --nb_zones ${zone} --random 1
        
        python train_model.py --data data/${OUTPUT_DATA_FILE} --output ${OUTPUT_DATA_FILE} &
      fi
    done
  done
done


# computation of ipca_reconstruction
ipca_batch_size=55

for component in {10,15,20,25,30,35,45,50}; do
  python generate/generate_reconstructed_data.py --features ${ipca_metric} --params "${component},${ipca_batch_size}"

  for zone in {6,8,10,12}; do
    OUTPUT_DATA_FILE="${ipca_metric}_nb_zones_${zone}_N${component}_BS${ipca_batch_size}"

    if grep -xq "${OUTPUT_DATA_FILE}" "${file_path}"; then
      
      echo "IPCA model ${OUTPUT_DATA_FILE} already generated"
    
    else
    
      echo "Run computation for IPCA model ${OUTPUT_DATA_FILE}"

      python generate/generate_dataset.py --output data/${OUTPUT_DATA_FILE} --features ${ipca_metric} --renderer ${renderer} --scenes ${scenes} --params "${component},${ipca_batch_size}" --nb_zones ${zone} --random 1
      python train_model.py --data data/${OUTPUT_DATA_FILE} --output ${OUTPUT_DATA_FILE} &
    fi
  done
done


# computation of fast_ica_reconstruction

for component in {50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200}; do
  python generate/generate_reconstructed_data.py --features ${fast_ica_metric} --params "${component}"

  for zone in {6,8,10,12}; do
    OUTPUT_DATA_FILE="${fast_ica_metric}_nb_zones_${zone}_N${component}"

    if grep -xq "${OUTPUT_DATA_FILE}" "${file_path}"; then
      
      echo "Fast ICA model ${OUTPUT_DATA_FILE} already generated"
    
    else
    
      echo "Run computation for Fast ICA model ${OUTPUT_DATA_FILE}"

      python generate/generate_dataset.py --output data/${OUTPUT_DATA_FILE} --features ${fast_ica_metric} --renderer ${renderer} --scenes ${scenes} --params "${component}" --nb_zones ${zone} --random 1
      
      python train_model.py --data data/${OUTPUT_DATA_FILE} --output ${OUTPUT_DATA_FILE} &
    fi
  done
done

# RUN LATER
# compute using all transformation methods
ipca_batch_size=55

: '
for begin in {80,85,90,95,100,105,110}; do
  for end in {150,160,170,180,190,200}; do
    for ipca_component in {10,15,20,25,30,35,45,50}; do
      for fast_ica_component in {50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200}; do
        for zone in {6,8,10,12}; do
          OUTPUT_DATA_FILE="${svd_metric}_B${begin}_E${end}_${ipca_metric}__N${ipca_component}_BS${ipca_batch_size}_${fast_ica_metric}_N${fast_ica_component}_nb_zones_${zone}"

          if grep -xq "${OUTPUT_DATA_FILE}" "${file_path}"; then
            
            echo "Transformation combination model ${OUTPUT_DATA_FILE} already generated"
          
          else
          
            echo "Run computation for Transformation combination model ${OUTPUT_DATA_FILE}"

            params="${begin}, ${end} :: ${ipca_component}, ${ipca_batch_size} :: ${fast_ica_component}"

            python generate/generate_dataset.py --output data/${OUTPUT_DATA_FILE} --metric ${all_features} --renderer ${renderer} --scenes ${scenes} --params "${params}" --nb_zones ${zone} --random 1
            
            python train_model.py --data data/${OUTPUT_DATA_FILE} --output ${OUTPUT_DATA_FILE} &
          fi
        done
      done
    done
  done
done
'
