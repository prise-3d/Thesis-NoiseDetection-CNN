# Noise detection project

## Requirements

```bash
git clone --recursive https://github.com/prise-3d/Thesis-NoiseDetection-CNN.git XXXXX
```

```bash
pip install -r requirements.txt
```

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
- **static** Use static file to manage (such as z-buffer, normals card...)
  - Param definition: *Name of image of scene need to be in {sceneName}/static/xxxx.png*
  - Example: *"img.png"*

**__Example:__**
```bash
python generate/generate_dataset.py --output data/output_data_filename --features "svd_reconstruction, ipca_reconstruction, fast_ica_reconstruction" --renderer "maxwell" --scenes "A, D, G, H" --params "100, 200 :: 50, 10 :: 50" --nb_zones 10 --random 1
```


Then, train model using your custom dataset:
```bash
python train_model --data data/custom_dataset --output output_model_name
```

## Modules

This project contains modules:
- **modules/utils/config.py**: *Store all configuration information about the project and dataset information*
- **modules/utils/data.py**: *Usefull methods used for dataset*
- **modules/models/metrics.py**: *Usefull methods for performance comparisons*
- **modules/models/models.py**: *Generation of CNN model*
- **modules/classes/Transformation.py**: *Transformation class for more easily manage computation*

All these modules will be enhanced during development of the project

## License

[MIT](https://github.com/prise-3d/Thesis-NoiseDetection-CNN/blob/master/LICENSE)