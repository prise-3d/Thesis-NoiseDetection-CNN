from modules.config.cnn_config import *

# store all variables from cnn config
context_vars = vars()

# Custom config used for redefined config variables if necessary

# folders

## noisy_folder                    = 'noisy'
## not_noisy_folder                = 'notNoisy'
backup_model_folder             = 'models_backup'

# file or extensions

perf_prediction_model_path      = 'predictions_models_results.csv'
## post_image_name_separator       = '___'

# variables
perf_train_header_file          = "model_name;global_train_size;global_test_size;filtered_train_size;filtered_test_size;f1_train;f1_test;recall_train;recall_test;presicion_train;precision_test;acc_train;acc_test;roc_auc_train;roc_auc_test;\n"
perf_prediction_header_file    = "data;data_size;model_name;accucary;f1;recall;precision;roc;\n"

features_choices_labels         = ['static', 'svd_reconstruction', 'fast_ica_reconstruction', 'ipca_reconstruction', 'min_diff_filter']

# parameters

sub_image_size                  = (200, 200)
keras_epochs                    = 100
## keras_batch                     = 32
## val_dataset_size                = 0.2

keras_img_size                  = (96, 96)