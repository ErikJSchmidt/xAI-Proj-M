"""
This file contains custom CNN models/networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import os

"""
Copied from https://www.kaggle.com/code/shadabhussain/cifar-10-cnn-using-pytorch/notebook .
Equips the model with some utility functions for training and validation the model.
"""
class ImageClassificationBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential()
        self.model_type = 'ImageClassification'

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
        
    def save_model(self, save_model_dir):
        '''
        Saves a model to save_model_dir.
        '''
        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H:%M')
        torch.save(
            self.network.state_dict(),
            save_model_dir + '/' + self.model_type + '_' + timestamp
        )

class SkipBlock(nn.Module):
    '''
    A Res-Net-style skip-connection.
    '''
    def __init__(self, in_channels, out_channels, stride = 1):
        '''
        Args:
        -----
        in_channels (int) : Number of input channels.
        out_channels (int) : Number of output channels.
        stride (int) : Stride for the Conv2d layers.
        '''
        super().__init__()

        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = None

        # We could consider skipping other blocks
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        id = x
        out = self.block(x)

        if self.skip is not None:
            id = self.skip(x)
        
        out += id
        out = F.relu(out)
        return out

class ModularSkip(nn.Module):
    '''
    A Res-Net-style skip-connection that takes the to-be-skipped blocks as parameter.
    '''
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.in_channels = block[0].in_channels        # Find the in_channels from block, and then out_channels
        self.out_channels = block[-1].out_channels     # This is definitely going to break when you use this class wrong...

        self.skip = nn.Sequential()

        if self.in_channels != self.out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
                nn.BatchNorm2d(self.out_channels)
            )
        else:
            self.skip = None
    
    def forward(self, x):
        id = x
        out = self.block(x)

        if self.skip is not None:
            id = self.skip(x)
        
        out += id
        out = F.relu(out)
        return out


"""
Copied from https://www.kaggle.com/code/shadabhussain/cifar-10-cnn-using-pytorch/notebook .
Implements a simple CNN architecture of three conv+pool blocks followed by final fully connected layers.
"""
class Cifar10CnnModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.model_type = 'Cifar10CnnModel'
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

    # def save_model(self, save_model_dir):
    #     if not os.path.exists(save_model_dir):
    #         os.makedirs(save_model_dir)

    #     timestamp = datetime.now().strftime("%Y%m%d_%H:%M")
    #     torch.save(
    #         self.network.state_dict(),
    #         save_model_dir + "/Cifar10CnnModel_" + timestamp
    #     )



'''
Similar structure to 18 Layer CNN from ResNet paper.
'''
class Plain18Layer(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.model_type = 'Plain18Layer'
        self.network = nn.Sequential(
            # Conv1: Prepare by mapping to 16 feature maps
            nn.Conv2d(3,8, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),

            # Conv2:                                        Learnable params
            nn.Conv2d(8,16, kernel_size=3, padding=1, bias=False),     # 8*16*3*3 = 1152
            nn.ReLU(),
            nn.Conv2d(16,16, kernel_size=3, padding=1, bias=False),     # 16*16*3*3 = 2304
            nn.ReLU(),
            nn.Conv2d(16,16, kernel_size=3, padding=1, bias=False),     # 16*16*3*3 = 2304
            nn.ReLU(),
            nn.Conv2d(16,16, kernel_size=3, padding=1, bias=False),     # 16*16*3*3 = 2304
            nn.ReLU(),                                                  # --------------------
            # conv2 total = 8064

            nn.MaxPool2d(2, 2), # output: 16 x 16 x 16

            # Conv3:
            nn.Conv2d(16,32, kernel_size=3, padding=1, bias=False),     # 16*32*3*3 = 4608
            nn.ReLU(),
            nn.Conv2d(32,32, kernel_size=3, padding=1, bias=False),     # 32*32*3*3 = 9216
            nn.ReLU(),
            nn.Conv2d(32,32, kernel_size=3, padding=1, bias=False),     # 32*32*3*3 = 9216
            nn.ReLU(),
            nn.Conv2d(32,32, kernel_size=3, padding=1, bias=False),     # 32*32*3*3 = 9216
            nn.ReLU(),                                                  # --------------------
            # conv3 total = 32256

            nn.MaxPool2d(2, 2), # output: 32 x 8 x 8

            # Conv4:
            nn.Conv2d(32,64, kernel_size=3, padding=1, bias=False),     # 32*64*3*3 = 18432
            nn.ReLU(),
            nn.Conv2d(64,64, kernel_size=3, padding=1, bias=False),     # 64*64*3*3 = 36864
            nn.ReLU(),
            nn.Conv2d(64,64, kernel_size=3, padding=1, bias=False),     # 64*64*3*3 = 36864
            nn.ReLU(),
            nn.Conv2d(64,64, kernel_size=3, padding=1, bias=False),     # 64*64*3*3 = 36864
            nn.ReLU(),                                      # --------------------
            # conv4 total = 129024

            nn.MaxPool2d(2, 2), # output: 64 x 4 x 4

            # Conv5:
            nn.Conv2d(64,128, kernel_size=3, padding=1, bias=False),    # 64*128*3*3 = 73728
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),  # 128*128*3*3 = 147456
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),  # 128*128*3*3 = 147456
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),  # 128*128*3*3 = 147456
            nn.ReLU(),                                      # --------------------

            nn.Flatten(),
            nn.Linear(128*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))

    def forward(self, xb):
        return self.network(xb)

    # def save_model(self, save_model_dir):
    #     if not os.path.exists(save_model_dir):
    #         os.makedirs(save_model_dir)

    #     timestamp = datetime.now().strftime("%Y%m%d_%H:%M")
    #     torch.save(
    #         self.network.state_dict(),
    #         save_model_dir + "/Plain18Layer" + timestamp
    #     )

'''
Model with 12 Convolution Layers, uniformly distributed across pooling steps.
'''
class Uniform12Layer(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.model_type = 'Uniform12Layer'
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, xb):
        return self.network(xb)


class Uniform12LayerSkipped(ImageClassificationBase):
    '''
    Model with 12 Convolution Layers, uniformly distributed across pooling steps + Skip Connections.
    '''
    def __init__(self):
        super().__init__()
        self.model_type = 'Uniform12LayerSkipped'
        self.network = nn.Sequential(
            SkipBlock(3, 32, stride = 1),
            nn.ReLU(),
            SkipBlock(32, 64, stride = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            SkipBlock(64, 96, stride = 1),
            nn.ReLU(),
            SkipBlock(96, 128, stride = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            SkipBlock(128, 192, stride = 1),
            nn.ReLU(),
            SkipBlock(192, 256, stride = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        return self.network(x)


'''
Model with 12 Convolution Layers, distributed toward the later pooling steps.
'''
class BackLoaded12Layer(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.model_type = 'BackLoaded12Layer'
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride= 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, xb):
        return self.network(xb)


class BackLoaded12LayerSkipped(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.model_type = 'BackLoaded12LayerSkipped'
        self.network = nn.Sequential(
            ModularSkip(
                nn.Sequential(
                    nn.Conv2d(3, 16, kernel_size=3, padding=1, bias = False),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
                )
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            ModularSkip(
                nn.Sequential(
                    nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.ReLU(),
                    nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.ReLU(),
                    nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1, bias=False),
                )
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            ModularSkip(
                nn.Sequential(
                    nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.ReLU(),
                    nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1, bias=False),
                )
            ),
            nn.ReLU(),
            ModularSkip(
                nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.ReLU(),
                    nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1, bias=False),
                )
            ),
            nn.ReLU(),
            ModularSkip(
                nn.Sequential(
                    nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.ReLU(),
                    nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.ReLU(),
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
                )
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        return self.network(x)


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

def fit_dyn(lim, min_epochs, lr, model, train_loader, val_loader, opt_func = torch.optim.SGD):
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
        if len(val_accs) >= 3 and (val_accs[-1] - val_accs[-3]) < lim and epoch > min_epochs:
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