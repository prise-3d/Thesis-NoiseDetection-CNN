# Noise detection with CNN

## Requirements

```bash
git clone --recursive https://github.com/prise-3d/Thesis-NoiseDetection-CNN.git
```

```bash
pip install -r requirements.txt
```

## Project structure

### Link to your dataset

You have to create a symbolic link to your own database which respects this structure:

- dataset/
  - Scene1/
    - zone00/
    - ...
    - zone15/
      - seuilExpe (file which contains threshold samples of zone image perceived by human)
    - Scene1_00050.png
    - Scene1_00070.png
    - ...
    - Scene1_01180.png
    - Scene1_01200.png
  - Scene2/
    - ...
  - ...

Create your symbolic link:

```
ln -s /path/to/your/data dataset
```

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
- **static** Use static file to manage (such as z-buffer, normals card...)
  - Param definition: *Name of image of scene need to be in {sceneName}/static/xxxx.png*
  - Example: *"img.png"*

**__Example:__**
```bash
python generate/generate_dataset.py --output data/output_data_filename --features "svd_reconstruction, ipca_reconstruction, fast_ica_reconstruction" --renderer "maxwell" --scenes "A, D, G, H" --params "100, 200 :: 50, 10 :: 50" --nb_zones 10 --random 1
```


Then, train model using your custom dataset:
```bash
python train_model.py --data data/custom_dataset --output output_model_name
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


## Use with Calculco (OAR service)

The `oar.example.sh` is an example of script to run in OAR platform.

```
oarsub -S oar.sh
```

Check your JOB_ID
```
oarstat
```

**Note:** Not forget to create symbolic link where it's necessary to logs results

```
ln -s /where/to/store/you/data data
ln -s /where/to/store/you/results/ results
ln -s /where/to/store/you/models_info models_info
ln -s /where/to/store/you/saved_models saved_models
```

or simply use this script:
```
bash generate_symlinks.sh /where/to/store/you
```

## License

[MIT](https://github.com/prise-3d/Thesis-NoiseDetection-CNN/blob/master/LICENSE)