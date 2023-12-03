import torch
import torch.nn as nn
import torch.nn.functional as F
from device_utils import get_default_device, to_device, accuracy


class ModelWrapper:

    def __init__(self, model: nn.Module):
        self.model = to_device(model, get_default_device())

    def training_step(self, batch):
        self.model.train()
        images, labels = batch
        out = self.model(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)
        return {'batch_loss': loss, 'batch_acc': acc}

    def validation_step(self, batch):
        #print("\n\n\n------------batch --------------\n" + str(batch))
        self.model.eval()
        #print("batch lenght" + str(len(batch)))
        images = batch[0]
        labels = batch[1]
        #print(images)
        #print(labels)
        #images, labels = batch
        out = self.model(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'batch_loss': loss.detach(), 'batch_acc': acc}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

    @torch.no_grad()
    def evaluate_model(self, data_loader):
        """
        :param data_loader: torch.utils.data.dataloader.DataLoader that provides the samples to evaluate on
        :return: {
            'loss': <mean over loss per batch>,
            'acc': <mean over accuracy per batch>
            }
        """
        # Per batch in data loader get mean metrics on that batch
        #batch_outputs = [self.validation_step(batch) for batch in data_loader]

        batch_outputs = []
        for batch in data_loader:
            batch_output = self.validation_step(batch)
            batch_outputs.append(batch_output)

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


