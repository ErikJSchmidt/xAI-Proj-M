import json
import sys
import os
from torchvision.datasets.utils import download_url
import tarfile

import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from model_plain_18_layer import Plain18Layer
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
        'model_name': trainer_config['model_name'],
        'store_model_dir': working_dir + trainer_config['store_model_dir_rel_path'],
        'train_dataset_dir': working_dir + trainer_config['train_dataset_dir_rel_path'],
        'test_dataset_dir': working_dir + trainer_config['test_dataset_dir_rel_path'],
        'random_seed': trainer_config['random_seed'],
        'batch_size': trainer_config['batch_size'],
        'epochs': trainer_config['epochs'],
        'learning_rate': trainer_config['learning_rate'],
        'weight_decay': trainer_config['weight_decay'],
        'momentum': trainer_config['momentum']
    }

    if trainer_config['model_name'] == "Plain18Layer":
        raw_model = Plain18Layer()
    else:
        raw_model = Plain18Layer()

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
