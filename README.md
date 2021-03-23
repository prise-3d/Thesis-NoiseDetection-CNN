# Noise detection with CNN

## Requirements

```bash
git clone --recursive https://github.com/prise-3d/Thesis-NoiseDetection-CNN.git
```

```bash
pip install -r requirements.txt
```

## Project structure

### Code architecture description

- **modules/\***: contains all modules usefull for the whole project (such as configuration variables)
- **analysis/\***: contains all jupyter notebook used for analysis during thesis
- **generate/\***: contains python scripts for generate data from scenes (described later)
- **prediction/\***: all python scripts for predict new threshold from computed models
- **simulation/\***: contains all bash scripts used for run simulation from models
- **display/\***: contains all python scripts used for display Scene information (such as Singular values...)
- **run/\***: bash scripts to run few step at once : 
  - generate custom dataset
  - train model
  - keep model performance
  - run simulation (if necessary)
- **others/\***: folders which contains others scripts such as script for getting performance of model on specific scene and write it into Mardown file.
- **custom_config.py**: override the main configuration project of `modules/config/global_config.py`
- **train_model.py**: script which is used to run specific model available.
- **prediction_model.py**: script which is used to run specific model with data in order to predict.

### Generated data directories:

- **data/\***: folder which will contain all generated *.train* & *.test* files in order to train model.
- **saved_models/\***: all scikit learn or keras models saved.
- **models_info/\***: all markdown files generated to get quick information about model performance and prediction obtained after running `run/runAll_*.sh` script.
- **results/**:  This folder contains `model_comparisons.csv` file used for store models performance.

## How to use

Generate reconstructed data from specific method of reconstruction (run only once time or clean data folder before):
```
python generate/generate_reconstructed_data.py -h
```

Generate custom dataset from one reconstructed method or multiples (implemented later)
```
python generate/generate_dataset.py -h
```

### Reconstruction parameter (--params)

List of expected parameter by reconstruction method:
- **svd_reconstruction:** Singular Values Decomposition
  - Param definition: *interval data used for reconstruction (begin, end)*
  - Example: *"100, 200"*
- **ipca_reconstruction:** Iterative Principal Component Analysis
  - Param definition: *number of components used for compression and batch size*
  - Example: *"30, 35"*
- **fast_ica_reconstruction:**  Fast Iterative Component Analysis
  - Param definition: *number of components used for compression*
  - Example: *"50"*
- **diff_filter:**  Bilateral diff filter
  - Param definition: *window size expected*
  - Example: *"5, 5"*
- **sobel_based_filter** Sobel based filter
  - Param definition: *K window size and pixel limite to remove*
  - Example: *"3, 30"*
- **static** Use static file to manage (such as z-buffer, normals card...)
  - Param definition: *Name of image of scene need to be in {sceneName}/static/xxxx.png*
  - Example: *"img.png"*

**__Example:__**
```bash
python generate/generate_dataset_sequence_file.py --output data/output_data_filename --folder <generated_data_folder> --features "svd_reconstruction, ipca_reconstruction, fast_ica_reconstruction" --params "100, 200 :: 50, 10 :: 50" --sequence 5 --size "100, 100" --selected_zones <zones_files.csv>
```


Then, train model using your custom dataset:
```bash
python train_lstm_model.py --train data/custom_dataset.train --test data/custom_dataset.test --chanels "1,3,3" --epochs 30 --batch_size 64 --seq_norm 1 --output output_model_name
```

### Predict image using model

Now we have a model trained, we can use it with an image as input:

```bash
python prediction/predict_noisy_image.py --image path/to/image.png --model saved_models/xxxxxx.json --features 'svd_reconstruction' --params '100, 200'
```

- **features**: feature choices need to be one of the listed above.

The model will return only 0 or 1:
- 1 means noisy image is detected.
- 0 means image seem to be not noisy.

### Simulate model on scene

All scripts named **prediction/predict_seuil_expe\*.py** are used to simulate model prediction during rendering process.

Once you have simulation done. Checkout your **threshold_map/%MODEL_NAME%/simulation\_curves\_zones\_\*/** folder and use it with help of **display_simulation_curves.py** script.

## License

[MIT](https://github.com/prise-3d/Thesis-NoiseDetection-CNN/blob/master/LICENSE)