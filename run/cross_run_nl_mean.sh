metric="nl_mean_noise_mask"
scenes="A,B,D,G,H,I"

all_scenes="A,B,C,D,E,F,G,H,I"

# file which contains model names we want to use for simulation
file_path="results/models_comparisons.csv"
stride=1
dist_patch=6

# for kernel in {3,5,7}; do
#     echo python generate/generate_reconstructed_data.py --features ${metric} --params ${kernel},${dist_patch} --size 100,100 --scenes ${all_scenes} --replace 0
# done

for scene in {"A","B","D","G","H","I"}; do

    # remove current scene test from dataset
    s="${scenes//,${scene}}"
    s="${s//${scene},}"

    for zone in {10,11,12}; do
        for kernel in {3,5,7}; do
            for balancing in {0,1}; do
            
                OUTPUT_DATA_FILE="${metric}_nb_zones_${zone}_W${window}_K${kernel}_balancing${balancing}_without_${scene}"
                OUTPUT_DATA_FILE_TEST="${metric}_nb_zones_${zone}_W${window}_K${kernel}_balancing${balancing}_scene_${scene}"

                if grep -q "${OUTPUT_DATA_FILE}" "${file_path}"; then
                
                    echo "SVD model ${OUTPUT_DATA_FILE} already generated"

                else

                    #echo "Run computation for SVD model ${OUTPUT_DATA_FILE}"
                    echo python generate/generate_dataset.py --output data/${OUTPUT_DATA_FILE_TEST} --features ${metric} --scenes ${scene} --params ${kernel},${dist_patch} --nb_zones ${zone} --random 1 --size 200,200     

                    echo python generate/generate_dataset.py --output data/${OUTPUT_DATA_FILE} --features ${metric} --scenes ${s} --params ${kernel},${dist_patch} --nb_zones ${zone} --random 1 --size 200,200     
                    
                    echo python train_model.py --data data/${OUTPUT_DATA_FILE} --output ${OUTPUT_DATA_FILE} --balancing ${balancing}
                    echo python prediction_model.py --data data/${OUTPUT_DATA_FILE_TEST}.train --model saved_models/${OUTPUT_DATA_FILE}.json
                fi 
            done
        done
    done
done

