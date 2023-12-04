import torch.nn as nn
from . import nn_modules as nnm
class Skipped32Layer(nn.Module):
    """"
    Start from Plain18Layer.
    Let Conv4 stay at 32 feature maps instead of 64 => less params per conv layer
        => more conv layers with overall same parameter count as Plain18LAyer
    31 conv layer + one fully connected => 32 layer network
    """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # Conv1: Prepare by mapping to 16 feature maps
            nn.Conv2d(3,16, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),

            # Conv2:                                                                           Learnable params
            nnm. ModularSkip(
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
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),                                                                        # --------------------
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
                    nn.Conv2d(32,32, kernel_size=3, padding=1, bias=False),     # 32*32*3*3 = 9216
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(32,32, kernel_size=3, padding=1, stride=2, bias=False),     # 32*32*3*3 = 9216
                )
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),                                                  # --------------------
            # conv3 total = 46080

            # output: 32 x 8 x 8

            # Conv4_a:
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
                    nn.Conv2d(32,32, kernel_size=3, padding=1, bias=False),     # 32*32*3*3 = 9216
                )
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),                                                                  #------------------
            # conv4_b total = 36864

            # Conv4_b:
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
                    nn.Conv2d(32,32, kernel_size=3, padding=1, bias=False),     # 32*32*3*3 = 9216
                )
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),     #------------------
            # conv4_b total = 36864

            # Conv4_c
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
                    nn.Conv2d(32,32, kernel_size=3, padding=1, bias=False),     # 32*32*3*3 = 9216
                )
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),                                                                    #------------------
            # conv4_c total = 36864

            # Conv4_d
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
            nn.Conv2d(32,128, kernel_size=3, padding=1, stride=2, bias=False),     # 32*128*3*3 = 36864
            nn.BatchNorm2d(128),
            nn.ReLU(),                                                                  #------------------
            # conv4_d total = 73728
            #=====================
            # conv4 total = 36864 + 36864 + 36864 + 73728 = 184320

            # output: 32 * 4 * 4

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
                    nn.Conv2d(128,128, kernel_size=3, padding=1, bias=False),    # 128*128*3*3 = 147456
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),  # 128*128*3*3 = 147456
                )
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),                                      # --------------------
            # conv5 total = 589824


            # Classification layers
            nn.AvgPool2d(4, stride=2),
            nn.Flatten(),
            nn.Linear(128, 10),
            nn.Softmax(1),
        )


    def forward(self, xb):
        return self.network(xb)