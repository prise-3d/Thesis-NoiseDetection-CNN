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

### Reconstruction parameter

List of expected parameter by reconstruction method:
- **svd:** Singular Values Decomposition
  - Param definition: *interval data used for reconstruction (begin, end)*
  - Example: *"100, 200"*
- **ipca:** Iterative Principal Component Analysis
  - Param definition: *number of components used for compression and batch size*
  - Example: *"50, 32"*
- **fast_ica:**  Fast Iterative Component Analysis
  - Param definition: *number of components used for compression*
  - Example: *"50"*

## Modules

This project contains modules:
- **modules/utils/config.py**: *Store all configuration information about the project and dataset information*
- **modules/utils/data.py**: *Usefull methods used for dataset*

All these modules will be enhanced during development of the project

## License

[MIT](https://github.com/prise-3d/Thesis-NoiseDetection-CNN/blob/master/LICENSE)