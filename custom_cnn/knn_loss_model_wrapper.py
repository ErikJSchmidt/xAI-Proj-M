import torch
import torch.nn as nn
from utility_functions import get_default_device, to_device, accuracy


class KnnLossModelWrapper:
    """
    The idea of a training run performed with this model wrapper is to train the model in different phases.
    The goal is to spread out the embeddings of, so that the centroids of different classes are further away.

    Training phases:
     1. Train with cross entropy
     2. Train with combination of centroid maximizing and class coherence loss

    """

    def __init__(self, model: nn.Module):
        self.model = to_device(model, get_default_device())

    def forward_batch(self, batch):
        self.model.train()
        images, labels = batch
        out = self.model(images)  # Generate predictions

        return out

    def training_step(self, batch, loss_func):
        self.model.train()
        images, labels = batch
        out = self.model(images)  # Generate predictions
        loss = loss_func(out, labels)  # Calculate loss
        acc = accuracy(out, labels)
        return {'batch_loss': loss, 'batch_acc': acc}

    def validation_step(self, batch, loss_func):
        self.model.eval()
        images = batch[0]
        labels = batch[1]
        out = self.model(images)  # Generate predictions
        loss = loss_func(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'batch_loss': loss.detach(), 'batch_acc': acc}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

    @torch.no_grad()
    def evaluate_model(self, data_loader, loss_func):
        """
        :param data_loader: torch.utils.data.dataloader.DataLoader that provides the samples to evaluate on
        :return: {
            'loss': <mean over loss per batch>,
            'acc': <mean over accuracy per batch>
            }
        """
        # Per batch in data loader get mean metrics on that batch
        batch_outputs = [self.validation_step(batch, loss_func) for batch in data_loader]

        """
        batch_outputs = []
   
        for batch in data_loader[0]:
            batch_output = self.validation_step(batch)
            batch_outputs.append(batch_output)
        """
        # Return list of batch results
        batch_losses = [x['batch_loss'] for x in batch_outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['batch_acc'] for x in batch_outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'mean_loss': epoch_loss.item(), 'mean_acc': epoch_acc.item()}

    def save_model(self, dir_path):
        torch.save(
            self.model.network.state_dict(),
            dir_path + "/model_state_dict"
        )

    def load_model_weights(self, model_path):
        self.model.network.load_state_dict(
            torch.load(model_path, map_location=torch.device(get_default_device())))