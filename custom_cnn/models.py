"""
This file contains custom CNN models/networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import os



"""
Utility
"""
class PrintLayer(nn.Module):
    def __init__(self, msg):
        super(PrintLayer, self).__init__()
        self.msg = msg

    def forward(self, x):
        # Do your print / debug stuff here
        print(self.msg)
        print(x.shape)
        return x

"""
Copied from https://www.kaggle.com/code/shadabhussain/cifar-10-cnn-using-pytorch/notebook .
Equips the model with some utility functions for training and validation the model.
"""
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


"""
Copied from https://www.kaggle.com/code/shadabhussain/cifar-10-cnn-using-pytorch/notebook .
Implements a simple CNN architecture of three conv+pool blocks followed by final fully connected layers.
"""
class Cifar10CnnModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

            nn.Flatten(),
            nn.Linear(256*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))

    def forward(self, xb):
        return self.network(xb)

    def save_model(self, save_model_dir):
        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H:%M")
        torch.save(
            self.network.state_dict(),
            save_model_dir + "/Cifar10CnnModel_" + timestamp
        )



'''
Similar structure to 18 Layer CNN from ResNet paper.


'''
class Plain18Layer(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # Conv1: Prepare by mapping to 16 feature maps
            nn.Conv2d(3,8, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),

            # Conv2:                                                                           Learnable params
            nn.Conv2d(8,16, kernel_size=3, padding=1, bias=False),      # 8*16*3*3 = 1152
            nn.ReLU(),
            nn.Conv2d(16,16, kernel_size=3, padding=1, bias=False),     # 16*16*3*3 = 2304
            nn.ReLU(),
            nn.Conv2d(16,16, kernel_size=3, padding=1, bias=False),     # 16*16*3*3 = 2304
            nn.ReLU(),
            nn.Conv2d(16,16, kernel_size=3, padding=1, stride=2, bias=False),     # 16*16*3*3 = 2304
            nn.ReLU(),                                                                        # --------------------
                                                                                            # conv2 total = 8064
            # output: 16 x 16 x 16

            # Conv3:
            nn.Conv2d(16,32, kernel_size=3, padding=1, bias=False),     # 16*32*3*3 = 4608
            nn.ReLU(),
            nn.Conv2d(32,32, kernel_size=3, padding=1, bias=False),     # 32*32*3*3 = 9216
            nn.ReLU(),
            nn.Conv2d(32,32, kernel_size=3, padding=1, bias=False),     # 32*32*3*3 = 9216
            nn.ReLU(),
            nn.Conv2d(32,32, kernel_size=3, padding=1, stride=2, bias=False),     # 32*32*3*3 = 9216
            nn.ReLU(),                                                  # --------------------
            # conv3 total = 32256

            # output: 32 x 8 x 8

            # Conv4:
            nn.Conv2d(32,64, kernel_size=3, padding=1, bias=False),     # 32*64*3*3 = 18432
            nn.ReLU(),
            nn.Conv2d(64,64, kernel_size=3, padding=1, bias=False),     # 64*64*3*3 = 36864
            nn.ReLU(),
            nn.Conv2d(64,64, kernel_size=3, padding=1, bias=False),     # 64*64*3*3 = 36864
            nn.ReLU(),
            nn.Conv2d(64,64, kernel_size=3, padding=1, stride=2, bias=False),     # 64*64*3*3 = 36864
            nn.ReLU(),                                      # --------------------
            # conv4 total = 129024

            # output: 64 x 4 x 4

            # Conv5:
            nn.Conv2d(64,128, kernel_size=3, padding=1, bias=False),    # 64*128*3*3 = 73728
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),  # 128*128*3*3 = 147456
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),  # 128*128*3*3 = 147456
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),  # 128*128*3*3 = 147456
            nn.ReLU(),                                      # --------------------
            # conv5 total = 516696

            #PrintLayer("Before AVG Pool"),

            # Classification layers
            nn.AvgPool2d(4, stride=2),
            #PrintLayer("After Pool"),
            nn.Flatten(),
            #PrintLayer("After flatten"),
            nn.Linear(128, 10),
            #PrintLayer("After linear"),
            nn.Softmax(),
            #PrintLayer("After softmax"),
        )



    def forward(self, xb):
        return self.network(xb)

    def save_model(self, save_model_dir):
        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H:%M")
        torch.save(
            self.network.state_dict(),
            save_model_dir + "/Plain18Layer" + timestamp
        )



def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        print("Epoce:", epoch)
        # Training Phase
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history

def fit_dyn(lim, lr, model, train_loader, val_loader, opt_func = torch.optim.SGD):
    history = []
    val_accs = []
    optimizer = opt_func(model.parameters(), lr)
    cont = True
    epoch = 1
    while cont:
        print('Epoch:', epoch)
        # Training Phase
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation Phase
        result = evaluate(model, val_loader)
        val_accs.append(result['val_acc'])
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
        if len(val_accs) >= 3 and (val_accs[-1] - val_accs[-3]) < lim:
            cont = False
        epoch += 1
    return history

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
#%%
