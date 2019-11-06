min_diff_metric="min_diff_filter"
svd_metric="svd_reconstruction"
ipca_metric="ipca_reconstruction"
fast_ica_metric="fast_ica_reconstruction"

scenes="A,B,D,G,H,I"

all_scenes="A,B,C,D,E,F,G,H,I"

# file which contains model names we want to use for simulation
file_path="results/models_comparisons.csv"
stride=1

# for window in {"3","5","7","9"}; do
#     echo python generate/generate_reconstructed_data.py --features ${metric} --params ${window},${window},${stride} --size 100,100 --scenes ${all_scenes}
# done

for scene in {"A","B","D","G","H","I"}; do

    # remove current scene test from dataset
    s="${scenes//,${scene}}"
    s="${s//${scene},}"

    for zone in {10,11,12}; do
        for window in {"3","5","7","9"}; do
            for balancing in {0,1}; do
            
                OUTPUT_DATA_FILE="${min_diff_metric}_nb_zones_${zone}_W${window}_S${stride}_balancing${balancing}_without_${scene}"
                OUTPUT_DATA_FILE_TEST="${min_diff_metric}_nb_zones_${zone}_W${window}_S${stride}_balancing${balancing}_scene_${scene}"

                if grep -q "${OUTPUT_DATA_FILE}" "${file_path}"; then
                
                    echo "SVD model ${OUTPUT_DATA_FILE} already generated"

                else

                    #echo "Run computation for SVD model ${OUTPUT_DATA_FILE}"
                    echo python generate/generate_dataset.py --output data/${OUTPUT_DATA_FILE_TEST} --features ${min_diff_metric} --scenes ${scene} --params ${window},${window},${stride} --nb_zones ${zone} --random 1 --size 100,100     

                    echo python generate/generate_dataset.py --output data/${OUTPUT_DATA_FILE} --features ${min_diff_metric} --scenes ${s} --params ${window},${window},${stride} --nb_zones ${zone} --random 1 --size 100,100     
                    
                    echo python train_model.py --data data/${OUTPUT_DATA_FILE} --output ${OUTPUT_DATA_FILE} --balancing ${balancing}
                    echo python prediction_model.py --data data/${OUTPUT_DATA_FILE_TEST}.train --model saved_models/${OUTPUT_DATA_FILE}.json
                fi 
            done
        done
    done
done


# First compute svd_reconstruction

for scene in {"A","B","D","G","H","I"}; do

    # remove current scene test from dataset
    s="${scenes//,${scene}}"
    s="${s//${scene},}"

    for begin in {80,85,90,95,100,105,110}; do
        for end in {150,160,170,180,190,200}; do
            
            # echo python generate/generate_reconstructed_data.py --features ${svd_metric} --params ${begin},${end} --size 100,100 --scenes ${all_scenes}
        
            OUTPUT_DATA_FILE_TEST="${svd_metric}_scene_E_nb_zones_16_B${begin}_E${end}_scene_${scene}"
            echo python generate/generate_dataset.py --output data/${OUTPUT_DATA_FILE} --features ${svd_metric} --scenes ${scene} --params ${begin},${end} --nb_zones 16 --random 1 --size 100,100

            for zone in {10,11,12}; do
                for balancing in {0,1}; do
                
                    OUTPUT_DATA_FILE="${svd_metric}_nb_zones_${zone}_B${begin}_E${end}_balancing${balancing}_without_${scene}"

                    if grep -xq "${OUTPUT_DATA_FILE}" "${file_path}"; then
                        
                        echo "SVD model ${OUTPUT_DATA_FILE} already generated"
                    
                    else
                    
                        # echo "Run computation for SVD model ${OUTPUT_DATA_FILE}"

                        echo python generate/generate_dataset.py --output data/${OUTPUT_DATA_FILE} --features ${svd_metric} --scenes ${s} --params ${begin},${end} --nb_zones ${zone} --random 1 --size 100,100     
                        
                        echo python train_model.py --data data/${OUTPUT_DATA_FILE} --output ${OUTPUT_DATA_FILE} --balancing ${balancing}
                        echo python prediction_model.py --data data/${OUTPUT_DATA_FILE_TEST}.train --model saved_models/${OUTPUT_DATA_FILE}.json
                    fi
                done
            done
        done
    done
done


# computation of ipca_reconstruction
ipca_batch_size=55

for scene in {"A","B","D","G","H","I"}; do

    # remove current scene test from dataset
    s="${scenes//,${scene}}"
    s="${s//${scene},}"

    for component in {10,15,20,25,30,35,45,50}; do

        # echo python generate/generate_reconstructed_data.py --features ${ipca_metric} --params ${component},${ipca_batch_size} --size 100,100 --scenes ${all_scenes}
        
        OUTPUT_DATA_FILE_TEST="${ipca_metric}_scene_E_nb_zones_16_B${begin}_E${end}_scene_${scene}"
        echo python generate/generate_dataset.py --output data/${OUTPUT_DATA_FILE} --features ${ipca_metric} --scenes ${scene} --params ${component},${ipca_batch_size} --nb_zones 16 --random 1 --size 100,100

        for zone in {10,11,12}; do
            for balancing in {0,1}; do
                OUTPUT_DATA_FILE="${ipca_metric}_nb_zones_${zone}_N${component}_BS${ipca_batch_size}_balancing${balancing}_without_${scene}"

                if grep -xq "${OUTPUT_DATA_FILE}" "${file_path}"; then
                
                    echo "IPCA model ${OUTPUT_DATA_FILE} already generated"
                
                else
                
                    # echo "Run computation for IPCA model ${OUTPUT_DATA_FILE}"

                    echo python generate/generate_dataset.py --output data/${OUTPUT_DATA_FILE} --features ${ipca_metric} --scenes ${s} --params ${component},${ipca_batch_size} --nb_zones ${zone} --random 1 --size 100,100
                    
                    echo python train_model.py --data data/${OUTPUT_DATA_FILE} --output ${OUTPUT_DATA_FILE} --balancing ${balancing}
                    echo python prediction_model.py --data data/${OUTPUT_DATA_FILE_TEST}.train --model saved_models/${OUTPUT_DATA_FILE}.json

                fi
            done 
        done
    done
done


# computation of fast_ica_reconstruction
for scene in {"A","B","D","G","H","I"}; do

    # remove current scene test from dataset
    s="${scenes//,${scene}}"
    s="${s//${scene},}"
        
    for component in {50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200}; do

        # echo python generate/generate_reconstructed_data.py --features ${fast_ica_metric} --params ${component} --size 100,100 --scenes ${all_scenes}
        
        OUTPUT_DATA_FILE_TEST="${fast_ica_metric}_scene_E_nb_zones_16_B${begin}_E${end}_scene_${scene}"
        echo python generate/generate_dataset.py --output data/${OUTPUT_DATA_FILE} --features ${fast_ica_metric} --scenes ${scene} --params ${component} --nb_zones 16 --random 1 --size 100,100

        for zone in {10,11,12}; do
            for balancing in {0,1}; do

                OUTPUT_DATA_FILE="${fast_ica_metric}_nb_zones_${zone}_N${component}_without_${scene}"

                if grep -xq "${OUTPUT_DATA_FILE}" "${file_path}"; then
                
                    echo "Fast ICA model ${OUTPUT_DATA_FILE} already generated"
                
                else
                
                    # echo "Run computation for Fast ICA model ${OUTPUT_DATA_FILE}"

                    echo python generate/generate_dataset.py --output data/${OUTPUT_DATA_FILE} --features ${fast_ica_metric} --scenes ${s} --params ${component} --nb_zones ${zone} --random 1 --size 100,100
                    
                    echo python train_model.py --data data/${OUTPUT_DATA_FILE} --output ${OUTPUT_DATA_FILE} --balancing ${balancing}
                    echo python prediction_model.py --data data/${OUTPUT_DATA_FILE_TEST}.train --model saved_models/${OUTPUT_DATA_FILE}.json
                fi
            done
        done
    done
done
