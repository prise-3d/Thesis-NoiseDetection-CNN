metric="sobel_based_filter"
scenes="A,B,D,G,H,I"

all_scenes="A,B,C,D,E,F,G,H,I"

# file which contains model names we want to use for simulation
file_path="results/models_comparisons.csv"
stride=1
window=3

for pixel in {30,40,50,60,70,80}; do
    echo python generate/generate_reconstructed_data.py --features ${metric} --params ${window},${pixel} --size 100,100 --scenes ${all_scenes} --replace 0
done

for scene in {"A","B","D","G","H","I"}; do

    # remove current scene test from dataset
    s="${scenes//,${scene}}"
    s="${s//${scene},}"

    for zone in {10,11,12}; do
        for pixel in {30,40,50,60,70,80}; do
            for balancing in {0,1}; do
            
                OUTPUT_DATA_FILE="${metric}_nb_zones_${zone}_K${window}_P${pixel}_balancing${balancing}_without_${scene}"
                OUTPUT_DATA_FILE_TEST="${metric}_nb_zones_${zone}_K${window}_P${pixel}_balancing${balancing}_scene_${scene}"

                if grep -q "${OUTPUT_DATA_FILE}" "${file_path}"; then
                
                    echo "SVD model ${OUTPUT_DATA_FILE} already generated"

                else

                    #echo "Run computation for SVD model ${OUTPUT_DATA_FILE}"
                    echo python generate/generate_dataset.py --output data/${OUTPUT_DATA_FILE_TEST} --features ${metric} --scenes ${scene} --params ${window},${pixel} --nb_zones ${zone} --random 1 --size 100,100     

                    echo python generate/generate_dataset.py --output data/${OUTPUT_DATA_FILE} --features ${metric} --scenes ${s} --params ${window},${pixel} --nb_zones ${zone} --random 1 --size 100,100     
                    
                    echo python train_model.py --data data/${OUTPUT_DATA_FILE} --output ${OUTPUT_DATA_FILE} --balancing ${balancing}
                    echo python prediction_model.py --data data/${OUTPUT_DATA_FILE_TEST}.train --model saved_models/${OUTPUT_DATA_FILE}.json
                fi 
            done
        done
    done
done

