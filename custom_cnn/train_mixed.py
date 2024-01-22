"""
For training runs where some epochs train the whole model and others only touch weight up until embedding layer.

"""

import json
import sys
import os
from torchvision.datasets.utils import download_url
import tarfile

from networks.model_plain_18_layer import Plain18Layer
from networks.model_plain_32_layer import Plain32Layer
from networks.model_skipped_18_layer import Skipped18Layer
from networks.model_skipped_18_layer_for_embedding import Skipped18LayerForEmbbeding
from networks.model_skipped_32_layer import Skipped32Layer

from networks.model_backloaded_12_layer import BackLoaded12Layer
from networks.model_backloaded_12_layer_skipped import BackLoaded12LayerSkipped
from networks.model_uniform_12_layer import Uniform12Layer
from networks.model_uniform_12_layer_skipped import Uniform12LayerSkipped
from networks.model_frontloaded_12_layer import FrontLoaded12Layer
from networks.model_frontloaded_12_layer_skipped import FrontLoaded12LayerSkipped

from networks.knn_loss_model import KNNLossModel

from model_wrapper import ModelWrapper
from model_trainer import ModelTrainer
from knn_loss_model_wrapper import KnnLossModelWrapper
from knn_loss_model_trainer import KnnLossModelTrainer

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
        # Batch size of the DataLoaders. How may samples are fed to the network per iteration (training_step)
        'batch_size': trainer_config['batch_size'],
        # Number of epochs to train the model for
        'epochs': trainer_config['epochs'],
        # Learning rate for SGD to start with
        'learning_rate_start': trainer_config['learning_rate_start'],
        # Weight decay of SGD
        'weight_decay': trainer_config['weight_decay'],
        # Momentum of SGD
        'momentum': trainer_config['momentum'],
        # Parameter to steer how long a plateau in validation loss needs to be before learning rate is reduced to 1/10
        'lr_reduce_patience': trainer_config['lr_reduce_patience'],
        # loss function that is currently used for training
        'loss_function': trainer_config['loss_function'],
        # Schedule that determines what loss function to use in which epoch: divergence_loss, cross_entropy_loss
        'training_phases': trainer_config['training_phases']
    }

    model_folder_path = None

    for loss_function_key in absolute_trainer_config['training_phases'].keys():
        number_of_epochs = int(absolute_trainer_config['training_phases'][loss_function_key])
        print(f"Train with {loss_function_key} for {number_of_epochs} epochs")

        phase_trainer_config = absolute_trainer_config.copy()
        phase_trainer_config['epochs'] = number_of_epochs
        phase_trainer_config['loss_function'] = loss_function_key

        if model_folder_path == None:
            print("No model weiths. Start training form scratch.")
            raw_model = get_model(trainer_config['model_name'])
            model_wrapper = KnnLossModelWrapper(raw_model)
            model_trainer = KnnLossModelTrainer(
                model_wrapper=model_wrapper,
                absolute_trainer_config=phase_trainer_config
            )
            model_folder_path = model_trainer.train_model()
        else:
            print(f"Load model weights from {model_folder_path}")
            raw_model = get_model(trainer_config['model_name'])
            model_wrapper = KnnLossModelWrapper(raw_model)
            model_wrapper.load_model_weights(model_folder_path + "/model_state_dict")
            model_trainer = KnnLossModelTrainer(
                model_wrapper=model_wrapper,
                absolute_trainer_config=phase_trainer_config
            )
            model_folder_path = model_trainer.train_model()


def get_model(model_name):
    if model_name == "Plain18Layer":
        raw_model = Plain18Layer()
    elif model_name == "Plain32Layer":
        raw_model = Plain32Layer()
    elif model_name == "Skipped18Layer":
        raw_model = Skipped18Layer()
    elif model_name == "Skipped32Layer":
        raw_model = Skipped32Layer()
    elif model_name == "Skipped18LayerForEmbbeding":
        raw_model = Skipped18LayerForEmbbeding()
    elif model_name == "Uniform12Layer":
        raw_model = Uniform12Layer()
    elif model_name == "Uniform12LayerSkipped":
        raw_model = Uniform12LayerSkipped()
    elif model_name == "BackLoaded12Layer":
        raw_model = BackLoaded12Layer()
    elif model_name == "BackLoaded12LayerSkipped":
        raw_model = BackLoaded12LayerSkipped()
    elif model_name == 'FrontLoaded12Layer':
        raw_model = FrontLoaded12Layer()
    elif model_name == 'FrontLoaded12LayerSkipped':
        raw_model = FrontLoaded12LayerSkipped()
    elif model_name == 'KNNLossModel':
        raw_model = KNNLossModel()
    else:
        print(f"Model name {model_name} not valid.")
    return raw_model


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