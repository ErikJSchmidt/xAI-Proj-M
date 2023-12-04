import json
import sys
import os
from torchvision.datasets.utils import download_url
import tarfile

from model_plain_18_layer import Plain18Layer
from model_plain_32_layer import Plain32Layer
from model_wrapper import ModelWrapper
from model_trainer import ModelTrainer


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
        tar.extractall(path='./data')


def train(working_dir, trainer_config):
    absolute_trainer_config = {
        # Class name of the model to be trained
        'model_name': trainer_config['model_name'],
        # relative path from working directory (custom_cnn) to directory where training results and models are stored
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
        'lr_reduce_patience': trainer_config['lr_reduce_patience']
    }

    if trainer_config['model_name'] == "Plain18Layer":
        raw_model = Plain18Layer()
    elif trainer_config['model_name'] == "Plain32Layer":
        raw_model = Plain32Layer()
    else:
        print(f"Model name {trainer_config['model_name']} not valid.")

    model_wrapper = ModelWrapper(raw_model)
    model_trainer = ModelTrainer(
        model_wrapper=model_wrapper,
        absolute_trainer_config=absolute_trainer_config
    )

    model_trainer.train_model()


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
