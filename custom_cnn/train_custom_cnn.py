import sys
import os
import time
from torchvision.datasets.utils import download_url
import tarfile

import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from models import Cifar10CnnModel, accuracy, evaluate, fit, get_default_device, DeviceDataLoader, to_device



# ---- Downloading Cifar10 ----
def download_and_unpack_dataset(data_dir):
    if os.path.exists(data_dir + "/cifar10"):
        print("Data already downloaded")
        return
    # Download the dataset to data directory
    dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
    download_url(dataset_url, data_dir)
    print("downloaded the compressed dataset")
    time.sleep(5)
    print("start extracting the dataset")
    # Extract dataset from compressed file
    with tarfile.open(data_dir + '/cifar10.tgz', 'r:gz') as tar:
        tar.extractall(path='./data')

# ---- Training the cnn ----
def load_dataset_and_train_cnn(data_dir):
    # Setting parameters for training
    train_dataset_root_path = data_dir + "/cifar10/train"
    random_seed = 420
    torch.manual_seed(random_seed)
    batch_size = 128
    num_epochs = 10
    opt_func = torch.optim.Adam
    lr = 0.001

    # Load dataset folder into torch dataset
    train_dataset = ImageFolder(
        root=train_dataset_root_path,
        transform=ToTensor()
    )

    # Split train dataset for training and validation
    validation_dataset_size = 5000
    train_dataset_size = len(train_dataset) - validation_dataset_size
    train_subset, validation_subset = random_split(train_dataset, [train_dataset_size, validation_dataset_size])

    # Create DataLoaders
    train_data_loader = DataLoader(
        dataset=train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False  # should be True if cuda is available
    )
    validation_data_loader = DataLoader(
        dataset=validation_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False  # should be True if cuda is available
    )

    # Get ready for GPU with Cuda
    device = get_default_device()
    print("Detected device: ", device)
    device_aware_train_data_loader = DeviceDataLoader(train_data_loader, device)
    device_aware_validation_data_loader = DeviceDataLoader(validation_data_loader, device)

    # Actually train the model
    model = to_device(Cifar10CnnModel(), device)

    history = fit(num_epochs, lr, model, device_aware_train_data_loader, device_aware_validation_data_loader, opt_func)
    print(history)

    evaluation = evaluate(model, device_aware_validation_data_loader)
    print(evaluation)



"""
Run in colab:

    %%python3 train_custom_cnn.py "/content/drive/MyDrive/Github/{github_repo}/custom_cnn/data"
"""
if __name__ == "__main__":
    data_dir = sys.argv[1]
    print("download cifar10 dataset to " + data_dir)
    download_and_unpack_dataset(data_dir)
    print("start training function")
    load_dataset_and_train_cnn(data_dir)
