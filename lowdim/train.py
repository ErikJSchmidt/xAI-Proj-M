"""
For training runs where some epochs train the whole model and others only touch weight up until embedding layer.

"""

import json
import sys
import os
from torchvision.datasets.utils import download_url
import tarfile
from low_dim_model_wrapper import LowDimModelWrapper
from low_dim_model_trainer import LowDimModelTrainer

"""
Modified version of our train.py script. Here the model can be trained with one loss for some epochs, and then the 
obtained model is saved. Then the model weights are loaded and the model is trained with an other loss for some
epochs before being saved again (to another file).
"""


# ---- Downloading Cifar10 ----
def download_and_unpack_dataset(data_dir):
    if os.path.exists(data_dir + "/cifar10"):
        print("Data already downloaded")
        return
    # Download the dataset to data directory
    dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
    download_url(dataset_url, data_dir)
    print("downloaded the compressed dataset")
    print("start extracting the dataset")
    # Extract dataset from compressed file
    with tarfile.open(data_dir + '/cifar10.tgz', 'r:gz') as tar:
        tar.extractall(path=data_dir) # hat ins repo directory heruntergeladen


def train(working_dir, trainer_config):
    absolute_trainer_config = {
        # Class name of the model to be trained
        'model_name': trainer_config['model_name'],
        # relative path from working directory (custom_cnn) to directory where training results and networks are stored
        'store_model_dir': working_dir + trainer_config['store_model_dir_rel_path'],
        # relative path from working directory (custom_cnn) to directory to be loaded as ImageFolder of train data
        'train_dataset_dir': working_dir + trainer_config['train_dataset_dir_rel_path'],
        # relative path from working directory (custom_cnn) to directory to be loaded as ImageFolder of test data
        'test_dataset_dir': working_dir + trainer_config['test_dataset_dir_rel_path'],
        # Random seed used to split train data into train and validation
        'random_seed': trainer_config['random_seed'],
    }

    training_phase_configs = trainer_config['training_phases']

    model_folder_path = None

    for phase_config in training_phase_configs:

        print(f"Train with {phase_config['loss_function']} for {phase_config['epochs']} epochs")

        config_for_trainer_this_phase = {**absolute_trainer_config, **phase_config}

        if model_folder_path == None:
            print("No model weights. Start training form scratch.")
            model_wrapper = LowDimModelWrapper()
            model_trainer = LowDimModelTrainer(
                model_wrapper=model_wrapper,
                absolute_trainer_config=config_for_trainer_this_phase
            )
            model_folder_path = model_trainer.train_and_save_model()
        else:
            print(f"Load model weights from {model_folder_path}")
            model_wrapper = LowDimModelWrapper()
            model_wrapper.load_model_weights(model_folder_path + "/model_state_dict")
            model_trainer = LowDimModelTrainer(
                model_wrapper=model_wrapper,
                absolute_trainer_config=config_for_trainer_this_phase
            )
            model_folder_path = model_trainer.train_and_save_model()





"""
Run in colab:

    %%python3 train.py "/content/drive/MyDrive/Github/{github_repo}/custom_cnn"
"""
if __name__ == "__main__":
    root_dir = sys.argv[1]

    config_file = open(root_dir + "/trainer_config.json", 'r')
    config = json.load(config_file)
    config_file.close()

    print("download cifar10 dataset to " + root_dir + config['dataset_root_dir_rel_path'])
    download_and_unpack_dataset(root_dir + config['dataset_root_dir_rel_path'])

    print("start training function")
    train(root_dir, config)