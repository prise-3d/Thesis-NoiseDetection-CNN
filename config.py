import os


# Custom config used for redefined config variables if necessary

# folders

output_data_folder              = 'data'
output_data_generated           = os.path.join(output_data_folder, 'generated')
output_datasets                 = os.path.join(output_data_folder, 'datasets')
output_zones_learned            = os.path.join(output_data_folder, 'learned_zones')
output_models                   = os.path.join(output_data_folder, 'saved_models')
output_results_folder           = os.path.join(output_data_folder, 'results')

noisy_folder                    = 'noisy'
not_noisy_folder                = 'notNoisy'
backup_model_folder             = os.path.join(output_data_folder, 'models_backup')

# file or extensions

perf_prediction_model_path      = 'predictions_models_results.csv'
results_filename                = 'results.csv'
## post_image_name_separator       = '___'

# variables
perf_train_header_file          = "model_name;global_train_size;global_test_size;filtered_train_size;filtered_test_size;f1_train;f1_test;recall_train;recall_test;presicion_train;precision_test;acc_train;acc_test;roc_auc_train;roc_auc_test;\n"
perf_prediction_header_file    = "data;data_size;model_name;accucary;f1;recall;precision;roc;\n"

features_choices_labels         = ['static', 'svd_reconstruction', 'svd_reconstruction_dyn', 'fast_ica_reconstruction', 'ipca_reconstruction', 'min_diff_filter', 'sobel_based_filter','nl_mean_noise_mask', 'gini_map']

# parameters

sub_image_size                  = (200, 200)
keras_epochs                    = 30
## keras_batch                     = 32
## val_dataset_size                = 0.2

keras_img_size                  = (200, 200)

# parameters
scene_image_quality_separator     = '_'
scene_image_extension             = '.png'