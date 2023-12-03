import torch.nn as nn
class Plain18Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # Conv1: Prepare by mapping to 16 feature maps
            nn.Conv2d(3,8, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),

            # Conv2:                                                                           Learnable params
            nn.Conv2d(8,16, kernel_size=3, padding=1, bias=False),       # 8*16*3*3 = 1152
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,16, kernel_size=3, padding=1, bias=False),     # 16*16*3*3 = 2304
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,16, kernel_size=3, padding=1, bias=False),     # 16*16*3*3 = 2304
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,16, kernel_size=3, padding=1, stride=2, bias=False),     # 16*16*3*3 = 2304
            nn.BatchNorm2d(16),
            nn.ReLU(),                                                                        # --------------------
            # conv2 total = 8064
            # output: 16 x 16 x 16

            # Conv3:
            nn.Conv2d(16,32, kernel_size=3, padding=1, bias=False),     # 16*32*3*3 = 4608
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,32, kernel_size=3, padding=1, bias=False),     # 32*32*3*3 = 9216
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,32, kernel_size=3, padding=1, bias=False),     # 32*32*3*3 = 9216
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,32, kernel_size=3, padding=1, stride=2, bias=False),     # 32*32*3*3 = 9216
            nn.BatchNorm2d(32),
            nn.ReLU(),                                                  # --------------------
            # conv3 total = 32256

            # output: 32 x 8 x 8

            # Conv4:
            nn.Conv2d(32,64, kernel_size=3, padding=1, bias=False),     # 32*64*3*3 = 18432
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64, kernel_size=3, padding=1, bias=False),     # 64*64*3*3 = 36864
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64, kernel_size=3, padding=1, bias=False),     # 64*64*3*3 = 36864
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64, kernel_size=3, padding=1, stride=2, bias=False),     # 64*64*3*3 = 36864
            nn.BatchNorm2d(64),
            nn.ReLU(),                                      # --------------------
            # conv4 total = 129024

            # output: 64 x 4 x 4

            # Conv5:
            nn.Conv2d(64,128, kernel_size=3, padding=1, bias=False),    # 64*128*3*3 = 73728
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),  # 128*128*3*3 = 147456
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),  # 128*128*3*3 = 147456
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),  # 128*128*3*3 = 147456
            nn.BatchNorm2d(128),
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
            nn.Softmax(1),
            #PrintLayer("After softmax"),
        )



    def forward(self, xb):
        return self.network(xb)