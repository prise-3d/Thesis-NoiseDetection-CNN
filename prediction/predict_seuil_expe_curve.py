# main imports
import sys, os, argparse
import subprocess
import numpy as np

# image processing imports
from ipfml.processing.segmentation import divide_in_blocks
from PIL import Image

# model imports
from sklearn.externals import joblib

# modules imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
from modules.utils import data as dt

# parameters from config and others
scenes_path               = cfg.dataset_path
min_max_filename          = cfg.min_max_filename_extension
threshold_expe_filename   = cfg.seuil_expe_filename

threshold_map_folder      = cfg.threshold_map_folder
threshold_map_file_prefix = cfg.threshold_map_folder + "_"

zones                     = cfg.zones_indices
maxwell_scenes            = cfg.maxwell_scenes_names
features_choices          = cfg.features_choices_labels

simulation_curves_zones   = "simulation_curves_zones_"
tmp_filename              = '/tmp/__model__img_to_predict.png'

current_dirpath = os.getcwd()


def main():

    parser = argparse.ArgumentParser(description="Script which predicts threshold using specific keras model")

    parser.add_argument('--features', type=str, 
                                     help="list of features choice in order to compute data",
                                     default='svd_reconstruction, ipca_reconstruction',
                                     required=True)
    parser.add_argument('--params', type=str, 
                                    help="list of specific param for each metric choice (See README.md for further information in 3D mode)", 
                                    default='100, 200 :: 50, 25',
                                    required=True)
    parser.add_argument('--model', type=str, help='.json file of keras model', required=True)
    parser.add_argument('--renderer', type=str, 
                                      help='Renderer choice in order to limit scenes used', 
                                      choices=cfg.renderer_choices, 
                                      default='all', 
                                      required=True)

    args = parser.parse_args()

    p_features    = list(map(str.strip, args.features.split(',')))
    p_params     = list(map(str.strip, args.params.split('::')))
    p_model_file = args.model
    p_renderer   = args.renderer

    scenes_list = dt.get_renderer_scenes_names(p_renderer)

    scenes = os.listdir(scenes_path)

    print(scenes)

    # go ahead each scenes
    for id_scene, folder_scene in enumerate(scenes):

        # only take in consideration renderer scenes
        if folder_scene in scenes_list:

            print(folder_scene)

            scene_path = os.path.join(scenes_path, folder_scene)

            # get all images of folder
            scene_images = sorted([os.path.join(scene_path, img) for img in os.listdir(scene_path) if cfg.scene_image_extension in img])
            number_scene_image = len(scene_images)

            start_quality_image = dt.get_scene_image_quality(scene_images[0])
            end_quality_image   = dt.get_scene_image_quality(scene_images[-1])
            # using first two images find the step of quality used
            quality_step_image  = dt.get_scene_image_quality(scene_images[1]) - start_quality_image

            threshold_expes = []
            threshold_expes_found = []
            block_predictions_str = []

            # get zones list info
            for index in zones:
                index_str = str(index)
                if len(index_str) < 2:
                    index_str = "0" + index_str
                zone_folder = "zone"+index_str

                threshold_path_file = os.path.join(os.path.join(scene_path, zone_folder), threshold_expe_filename)

                with open(threshold_path_file) as f:
                    threshold = int(f.readline())
                    threshold_expes.append(threshold)

                    # Initialize default data to get detected model threshold found
                    threshold_expes_found.append(int(end_quality_image)) # by default use max

                block_predictions_str.append(index_str + ";" + p_model_file + ";" + str(threshold) + ";" + str(start_quality_image) + ";" + str(quality_step_image))

            # for each images
            for img_path in scene_images:

                current_img = Image.open(img_path)
                img_blocks = divide_in_blocks(current_img, cfg.keras_img_size)

                current_quality_image = dt.get_scene_image_quality(img_path)

                for id_block, block in enumerate(img_blocks):

                    # check only if necessary for this scene (not already detected)
                    #if not threshold_expes_detected[id_block]:

                        tmp_file_path = tmp_filename.replace('__model__',  p_model_file.split('/')[-1].replace('.json', '_'))
                        block.save(tmp_file_path)

                        python_cmd = "python predict_noisy_image.py --image " + tmp_file_path + \
                                        " --features " + p_features + \
                                        " --params " + p_params + \
                                        " --model " + p_model_file 

                        ## call command ##
                        p = subprocess.Popen(python_cmd, stdout=subprocess.PIPE, shell=True)

                        (output, err) = p.communicate()

                        ## Wait for result ##
                        p_status = p.wait()

                        prediction = int(output)

                        # save here in specific file of block all the predictions done
                        block_predictions_str[id_block] = block_predictions_str[id_block] + ";" + str(prediction)

                        print(str(id_block) + " : " + str(current_quality_image) + "/" + str(threshold_expes[id_block]) + " => " + str(prediction))

                print("------------------------")
                print("Scene " + str(id_scene + 1) + "/" + str(len(scenes)))
                print("------------------------")

            # end of scene => display of results

            # construct path using model name for saving threshold map folder
            model_threshold_path = os.path.join(threshold_map_folder, p_model_file.split('/')[-1].replace('.joblib', ''))

            # create threshold model path if necessary
            if not os.path.exists(model_threshold_path):
                os.makedirs(model_threshold_path)

            map_filename = os.path.join(model_threshold_path, simulation_curves_zones + folder_scene)
            f_map = open(map_filename, 'w')

            for line in block_predictions_str:
                f_map.write(line + '\n')
            f_map.close()

            print("Scene " + str(id_scene + 1) + "/" + str(len(maxwell_scenes)) + " Done..")
            print("------------------------")

            print("Model predictions are saved into %s" % map_filename)


if __name__== "__main__":
    main()
