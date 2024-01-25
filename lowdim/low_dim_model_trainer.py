import json

from low_dim_model_wrapper import LowDimModelWrapper
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


class LowDimModelTrainer:

    def __init__(self, model_wrapper: LowDimModelWrapper, absolute_trainer_config):
        self.model_wrapper: LowDimModelWrapper = model_wrapper
        self.trainer_config = absolute_trainer_config
        self.knn_loss = knn_loss.KNNLoss(
            classes=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        )

    def train_and_save_model(self):
        """

        :return: path to subfolder where the model weights and training history were stored
        """
        # Setting parameters for training
        torch.manual_seed(self.trainer_config['random_seed'])

        device_aware_train_data_loader, device_aware_validation_data_loader, device_aware_test_data_loader = self.prepare_dataloaders()

        print("ToDo: Why is device_validation_data_loader wrapped in a tuple?")
        print(type(device_aware_train_data_loader))
        print(type(device_aware_validation_data_loader))
        print(type(device_aware_test_data_loader))

        training_history = self.fit_and_store_results(
            epochs=self.trainer_config['epochs'],
            lr=self.trainer_config['learning_rate_start'],
            weight_decay=self.trainer_config['weight_decay'],
            momentum=self.trainer_config['momentum'],
            train_loader=device_aware_train_data_loader,
            val_loader=device_aware_validation_data_loader
        )

        print("Done with training. Now on to test set")
        test_set_result = self.evaluate_model(device_aware_test_data_loader)

        print(f"Done processing the test set")

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
            'final_test_set_results': test_set_result
        }

        # create chroma db to store embeddings of all epochs
        self.store_training_run_embeddings(training_run, model_subfolder_path)
        # store parameters of the training run
        training_run_file = open(model_subfolder_path + "/training_run.json", "w+")
        json.dump({
            'training_config': self.trainer_config,
            'epochs':[{'lr': epoch_result['lr']} for epoch_result in training_history]
        }, training_run_file)
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

    def fit_and_store_results(self, epochs, lr, weight_decay, momentum, train_loader, val_loader):
        """
        Fit the model in the model_wrapper with the loss function specified in the config
        :param epochs:
        :param lr:
        :param weight_decay:
        :param momentum:
        :param train_loader:
        :param val_loader:
        :return:
        """
        history = []
        optimizer = torch.optim.SGD(
            self.model_wrapper.model.parameters(),
            lr,
            weight_decay=weight_decay,
            momentum=momentum
        )

        loss_function_key = self.trainer_config["loss_function"]
        loss_func = self.get_loss_function_for_key(loss_function_key)
        print(f"Use loss function: {loss_function_key}, {str(loss_func)}")

        learning_rate_scheduler = ReduceLROnPlateau(optimizer, 'min',
                                                    patience=self.trainer_config['lr_reduce_patience'])

        for epoch in range(epochs):
            print("Epoche:", epoch)

            # Training Phase
            print("training")
            # the embeddings the model returned for this epoch
            train_embedding_batches = []
            # the output prediction the model returned for all embeddings in the epoch
            train_prediction_batches = []
            # labels in order as processed in this epoch
            train_label_batches = []

            train_losses = []
            for i, batch in enumerate(train_loader):
                batch_images, batch_labels = batch
                batch_embeddings, batch_out = self.model_wrapper.training_step(batch_images)
                if loss_function_key == "divergence_loss":
                    # The divergence loss is plugged directly onto the final embedding layer of the CNN and ignores the fc layer
                    batch_loss = loss_func(batch_embeddings, batch_labels)
                elif loss_function_key == "cross_entropy":
                    # The cross entropy loss is plugged onto the prob. dist. output of the fc layer
                    batch_loss = loss_func(batch_out, batch_labels)
                # if divergence loss only CNN weights should be adjusted, if cross entropy all weights
                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_embedding_batches.append(batch_embeddings)
                train_prediction_batches.append(batch_out)
                train_label_batches.append(batch_labels)
                train_losses.append(batch_loss)

            # Validation phase
            print("validation")
            val_embedding_batches = []
            val_prediction_batches = []
            val_label_batches = []
            val_losses = []
            with torch.no_grad():
                for i, batch in enumerate(val_loader[0]):
                    batch_images, batch_labels = batch
                    batch_embeddings, batch_out = self.model_wrapper.validation_step(batch_images)
                    if loss_function_key == "divergence_loss":
                        # The divergence loss is plugged directly onto the final embedding layer of the CNN and ignores the fc layer
                        batch_loss = loss_func(batch_embeddings, batch_labels)
                    elif loss_function_key == "cross_entropy":
                        # The cross entropy loss is plugged onto the prob. dist. output of the fc layer
                        batch_loss = loss_func(batch_out, batch_labels)

                    val_embedding_batches.append(batch_embeddings)
                    val_prediction_batches.append(batch_out)
                    val_label_batches.append(batch_labels)
                    val_losses.append(batch_loss)

            epoch_result = {
                'lr': optimizer.param_groups[0]['lr'],
                'train_results': {
                    'batch_losses': train_losses,
                    'embeddings': train_embedding_batches,
                    'predictions': train_prediction_batches,
                    'labels': train_label_batches
                },
                'val_results': {
                    'batch_losses': val_losses,
                    'embeddings': val_embedding_batches,
                    'predictions': val_prediction_batches,
                    'labels': val_label_batches
                }
            }

            # ToDo: store epoch results to folder

            avg_train_loss_of_epoch = torch.stack(train_losses).mean()
            learning_rate_scheduler.step(avg_train_loss_of_epoch)

            history.append(epoch_result)
            print(f"Epoch {epoch}:\navg loss: {avg_train_loss_of_epoch}")

        return history

    def evaluate_model(self, test_data_loader):
        """
        :param test_data_loader: torch.utils.data.dataloader.DataLoader that provides the samples to evaluate on
        :return: all embeddings, prediction and true labels that were fed through the model from the test_data_loader
        """
        with torch.no_grad():
            test_embedding_batches = []
            test_prediction_batches = []
            test_label_batches = []
            for batch in test_data_loader:
                batch_images, batch_labels = batch
                batch_embeddings, batch_out = self.model_wrapper.validation_step(batch_images)
                test_embedding_batches.append(batch_embeddings)
                test_prediction_batches.append(batch_out)
                test_label_batches.append(batch_labels)

            test_result = {
                'embeddings': test_embedding_batches,
                'predictions': test_prediction_batches,
                'labels': test_label_batches
            }
            return test_result

    def get_loss_function_for_key(self, key):
        if key == "divergence_loss":
            return self.knn_loss.divergence_loss
        elif key == "cross_entropy_loss":
            return F.cross_entropy

    def store_training_run_embeddings(self, training_run, model_subfoler_path):
        """
        training_run = {
            'training_config': self.trainer_config,
            'training_history': training_history,
            'final_test_set_results': test_set_result
        }
        """

        # store training and validation result of each epoch
        for epoch, epoch_result in enumerate(training_run['training_history']):
            # the table name under which the embeddings produced in this epoch get stored
            epoch_folder_path = f"{model_subfolder_path}/epoch_{epoch}"
            os.makedirs(epoch_folder_path)

            # store data on the training samples processed in this epoch
            epoch_train_embeddings = []
            epoch_train_predictions = []
            epoch_train_labels = []
            train_results = epoch_result['train_results']
            for batch_nr in range(0, len(train_results['batch_losses'])):
                # add training data embeddings calculated for this batch
                batch_loss = train_results['batch_losses'][batch_nr]
                batch_train_embeddings = train_results['embeddings'][batch_nr].tolist()
                batch_train_predictions = epoch_result['training_results']['predictions'][batch_nr].tolist()
                batch_labels = train_results['labels'][batch_nr]

                epoch_train_embeddings.extend(batch_train_embeddings)
                epoch_train_predictions.extend(batch_train_predictions)
                epoch_train_labels.extend(batch_labels)

            torch.save(epoch_train_embeddings, f"{epoch_folder_path}/train_embeddings.pt")
            torch.save(epoch_train_predictions, f"{epoch_folder_path}/train_predictions.pt")
            torch.save(epoch_train_labels, f"{epoch_folder_path}/train_labels.pt")

            # store data on the validation samples processed in this epoch
            val_results = epoch_result['val_results']
            epoch_val_embeddings = []
            epoch_val_predictions = []
            epoch_val_labels = []
            for batch_nr in range(0, len(val_results['batch_losses'])):
                # add validation data embeddings calculated for this batch
                batch_loss = val_results['batch_losses'][batch_nr]
                batch_val_embeddings = val_results['embeddings'][batch_nr].tolist()
                batch_val_predictions = val_results['predictions'][batch_nr]
                batch_labels = val_results['labels'][batch_nr]

                epoch_val_embeddings.extend(batch_val_embeddings)
                epoch_val_predictions.extend(batch_val_predictions)
                epoch_val_labels.extend(batch_labels)

            torch.save(epoch_val_embeddings, f"{epoch_folder_path}/val_embeddings.pt")
            torch.save(epoch_val_predictions, f"{epoch_folder_path}/val_predictions.pt")
            torch.save(epoch_val_labels, f"{epoch_folder_path}/val_labels.pt")


        # store results of the test set
        test_folder_path = f"{model_subfolder_path}/test"
        os.makedirs(test_folder_path)
        test_results = training_run['final_test_set_results']

        torch.save(test_results['embeddings'], f"{test_folder_path}/test_embeddings.pt")
        torch.save(test_results['predictions'], f"{test_folder_path}/test_perdictions.pt")
        torch.save(test_results['labels'], f"{test_folder_path}/test_labels.pt")
