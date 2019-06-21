# Noise detection project

## Requirements

```
pip install -r requirements.txt
```

## How to use

Generate reconstructed data from specific method of reconstruction (run only once time or clean data folder before):
```
python generate_reconstructed_data.py -h
```

Generate custom dataset from one reconstructed method or multiples (implemented later)
```
python generate_dataset.py -h
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

**__Example:__**
```bash
python generate_dataset_3D.py --output data/output_data_filename --metrics "svd_reconstruction, ipca_reconstruction, fast_ica_reconstruction" --renderer "maxwell" --scenes "A, D, G, H" --params "100, 200 :: 50, 10 :: 50" --nb_zones 10 --random 1
```

## Modules

This project contains modules:
- **modules/utils/config.py**: *Store all configuration information about the project and dataset information*
- **modules/utils/data.py**: *Usefull methods used for dataset*

All these modules will be enhanced during development of the project

## License

[MIT](https://github.com/prise-3d/Thesis-NoiseDetection-CNN/blob/master/LICENSE)