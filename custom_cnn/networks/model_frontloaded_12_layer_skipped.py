from torch import nn
from . import nn_modules as sb

'''
Model with 12 Convolution Layers, distributed toward the earlier pooling steps.
'''


class FrontLoaded12LayerSkipped(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_type = 'FrontLoaded12LayerSkipped'
        self.network = nn.Sequential(
            sb.ModularSkip(
                nn.Sequential(
                    nn.Conv2d(3, 16, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                )
            ),
            nn.ReLU(),
            sb.ModularSkip(
                nn.Sequential(
                    nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
                )
            ),
            nn.ReLU(),
            sb.ModularSkip(
                nn.Sequential(
                    nn.Conv2d(64, 96, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1),
                )
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            sb.ModularSkip(
                nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 192, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
                )
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            sb.ModularSkip(
                nn.Sequential(
                    nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
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