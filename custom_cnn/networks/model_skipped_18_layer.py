import torch.nn as nn
from  . import nn_modules as nnm
class Skipped18Layer(nn.Module):
    """"
    Oriented at plain 18 layer CNN from ResNet paper as close as possible.
    17 convolutional layers and one fully connected one => 18 Layer
    """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # Conv1: Prepare by mapping to 16 feature maps
            nn.Conv2d(3,16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),


            # Conv2:
            nnm.ModularSkip(
                nn.Sequential(
                    nn.Conv2d(16,16, kernel_size=3, padding=1, bias=False),       # 16*16*3*3 = 2304
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.Conv2d(16,16, kernel_size=3, padding=1, bias=False),     # 16*16*3*3 = 2304
                )
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nnm.ModularSkip(
                nn.Sequential(
                    nn.Conv2d(16,16, kernel_size=3, padding=1, bias=False),     # 16*16*3*3 = 2304
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.Conv2d(16,32, kernel_size=3, padding=1, stride=2, bias=False),     # 16*32*3*3 = 4608
                )
            ),                                                                       # --------------------
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # conv2 total = 11520
            # output: 32 x 16 x 16

            # Conv3:
            nnm.ModularSkip(
                nn.Sequential(
                    nn.Conv2d(32,32, kernel_size=3, padding=1, bias=False),     # 32*32*3*3 = 9216
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(32,32, kernel_size=3, padding=1, bias=False),     # 32*32*3*3 = 9216
                )
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nnm.ModularSkip(
                nn.Sequential(
                    nn.Conv2d(32,32, kernel_size=3, padding=1, bias=False),     # 32*32*3*3 = 9216
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(32,64, kernel_size=3, padding=1, stride=2, bias=False),     # 32*64*3*3 = 18432
                )
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),                                       # --------------------
            # conv3 total = 46080

            # output: 64 x 8 x 8

            # Conv4:
            nnm.ModularSkip(
                nn.Sequential(
                    nn.Conv2d(64,64, kernel_size=3, padding=1, bias=False),     # 64*64*3*3 = 36864
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(64,64, kernel_size=3, padding=1, bias=False),     # 64*64*3*3 = 36864
                )
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nnm.ModularSkip(
                nn.Sequential(
                    nn.Conv2d(64,64, kernel_size=3, padding=1, bias=False),     # 64*64*3*3 = 36864
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(64,128, kernel_size=3, padding=1, stride=2, bias=False),     # 64*128*3*3 = 73728
                )
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(), # --------------------
            # conv4 total = 184320

            # output: 128 x 4 x 4

            # Conv5:
            nnm.ModularSkip(
                nn.Sequential(
                    nn.Conv2d(128,128, kernel_size=3, padding=1, bias=False),    # 128*128*3*3 = 147456
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),  # 128*128*3*3 = 147456
                )
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nnm.ModularSkip(
                nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),  # 128*128*3*3 = 147456
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),  # 128*128*3*3 = 147456
                )
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),# --------------------
            # conv5 total = 589824


            # Classification layers
            nn.AvgPool2d(4, stride=2),
            nn.Flatten(),
            nn.Linear(128, 10),
            nn.Softmax(1),
        )



    def forward(self, xb):
        return self.network(xb)