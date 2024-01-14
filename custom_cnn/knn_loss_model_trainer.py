import json

from knn_loss_model_wrapper import KnnLossModelWrapper
import knn_loss
from datetime import datetime
import os
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utility_functions import DeviceDataLoader, get_default_device
import torch.nn.functional as F
from torch import nn


class KnnLossModelTrainer:

    def __init__(self, model_wrapper: KnnLossModelWrapper, absolute_trainer_config):
        self.model_wrapper = model_wrapper
        self.trainer_config = absolute_trainer_config
        self.knn_loss = knn_loss.KNNLoss(
            classes= torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        )

    def train_model(self):
        # Setting parameters for training
        torch.manual_seed(self.trainer_config['random_seed'])
        opt_func = torch.optim.SGD

        device_aware_train_data_loader, device_aware_validation_data_loader, device_aware_test_data_loader = self.prepare_dataloaders()

        print("ToDo: Why is device_validation_data_loader wrapped in a tuple?")
        print(type(device_aware_train_data_loader))
        print(type(device_aware_validation_data_loader))
        print(type(device_aware_test_data_loader))

        training_history = self.fit(
            epochs=self.trainer_config['epochs'],
            lr=self.trainer_config['learning_rate_start'],
            weight_decay=self.trainer_config['weight_decay'],
            momentum=self.trainer_config['momentum'],
            train_loader=device_aware_train_data_loader,
            val_loader=device_aware_validation_data_loader
        )


        print("Done with training. Now on to test set")
        final_test_result = self.model_wrapper.evaluate_model(device_aware_test_data_loader, F.cross_entropy)
        final_test_result = {
            'test_loss': final_test_result['mean_loss'],
            'test_acc': final_test_result['mean_acc']
        }
        print(f"Final test result:\n{final_test_result}")

        # Create subfolder for model and results of this training run
        if not os.path.exists(self.trainer_config['store_model_dir']):
            os.makedirs(self.trainer_config['store_model_dir'])

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        model_subfolder_name = self.trainer_config['model_name'] + "_" + timestamp
        model_subfolder_path = self.trainer_config['store_model_dir'] + "/" + model_subfolder_name
        os.makedirs(model_subfolder_path)

        # Save training results
        training_run = {
            'training_config': self.trainer_config,
            'training_history': training_history,
            'final_test_result': final_test_result
        }

        training_run_file = open(model_subfolder_path + "/training_run.json", "w+")
        json.dump(training_run, training_run_file)
        training_run_file.close()

        # Save the fitted model
        self.model_wrapper.save_model(
            dir_path=model_subfolder_path
        )

        return model_subfolder_path

    def prepare_dataloaders(self):
        print("Load train data from " + self.trainer_config['train_dataset_dir'])
        # Load dataset folder into torch dataset
        train_dataset = ImageFolder(
            root=self.trainer_config['train_dataset_dir'],
            transform=ToTensor()
        )

        # Split train dataset for training and validation
        validation_dataset_size = 5000
        train_dataset_size = len(train_dataset) - validation_dataset_size
        # ToDo Stratify
        train_subset, validation_subset = random_split(train_dataset, [train_dataset_size, validation_dataset_size])

        # Create DataLoaders
        train_data_loader = DataLoader(
            dataset=train_subset,
            batch_size=self.trainer_config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=False  # should be True if cuda is available
        )
        validation_data_loader = DataLoader(
            dataset=validation_subset,
            batch_size=self.trainer_config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=False  # should be True if cuda is available
        )

        print(
            "Load test data from " + self.trainer_config['test_dataset_dir'])  # Load dataset folder into torch dataset
        test_dataset = ImageFolder(
            root=self.trainer_config['train_dataset_dir'],
            transform=ToTensor()
        )
        # Create DataLoader
        test_data_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.trainer_config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=False  # should be True if cuda is available
        )

        # Get ready for GPU with Cuda
        device = get_default_device()
        print("Detected device: ", device)
        device_aware_train_data_loader = DeviceDataLoader(train_data_loader, device)
        device_aware_validation_data_loader = DeviceDataLoader(validation_data_loader, device),
        device_aware_test_data_loader = DeviceDataLoader(test_data_loader, device)

        return [
            device_aware_train_data_loader,
            device_aware_validation_data_loader,
            device_aware_test_data_loader
        ]

    def fit(self, epochs, lr, weight_decay, momentum, train_loader, val_loader):
        history = []
        optimizer = torch.optim.SGD(
            self.model_wrapper.model.parameters(),
            lr,
            weight_decay=weight_decay,
            momentum=momentum
        )

        learning_rate_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=self.trainer_config['lr_reduce_patience'])

        loss_function_key = self.trainer_config["loss_function"]
        loss_func = self.get_loss_function_for_key(loss_function_key)
        print(f"Use loss function: {loss_function_key}, {str(loss_func)}")

        if loss_function_key in ["divergence_loss"]:
            print("Remove fc layer before training")
            removed_fc = self.model_wrapper.model.fc
            self.model_wrapper.model.fc = Identity()

        for epoch in range(epochs):
            print("Epoche:", epoch)

            # Training Phase
            train_losses = []
            train_accuracies = []
            for i, batch in enumerate(train_loader):
                batch_train_result = self.model_wrapper.training_step(batch, loss_func=loss_func)
                loss = batch_train_result['batch_loss']
                train_losses.append(loss)
                train_accuracies.append(batch_train_result['batch_acc'])
                # print(type(loss))
                # print(loss.shape)
                # print(loss)
                # print(i)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            # Validation phase
            print("evaluate")
            epoch_val_result = self.model_wrapper.evaluate_model(val_loader[0], loss_func)

            epoch_result = {
                'lr': optimizer.param_groups[0]['lr'],
                'val_loss': epoch_val_result['mean_loss'],
                'val_acc': epoch_val_result['mean_acc'],
                'train_loss': torch.stack(train_losses).mean().item(),
                'train_acc': torch.stack(train_accuracies).mean().item(),
            }

            learning_rate_scheduler.step(epoch_result['val_acc'])

            history.append(epoch_result)
            print(f"Epoch {epoch}:\n{epoch_result}")

        if loss_function_key in ["divergence_loss"]:
            print("Add fc that was removed before back to model")
            self.model_wrapper.model.fc = removed_fc

        return history

    def get_loss_function_for_key(self, key):
        if key == "divergence_loss":
            return self.knn_loss.divergence_loss
        elif key == "cross_entropy_loss":
            return F.cross_entropy




class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x