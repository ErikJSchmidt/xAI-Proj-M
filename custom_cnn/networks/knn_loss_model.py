from torch import nn
from . import nn_modules as sb

class KNNLossModel(nn.Module):
    '''
    Model similar to Uniform12LayerSkipped, but ending in a 256D vector which is the embedding we attempt to learn.
    '''
    def __init__(self):
        super().__init__()
        self.model_type = 'KNNLossModel'
        self.network = nn.Sequential(
            sb.SkipBlock(3, 32, stride = 1),
            nn.ReLU(),
            sb.SkipBlock(32, 64, stride = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            sb.SkipBlock(64, 96, stride = 1),
            nn.ReLU(),
            sb.SkipBlock(96, 128, stride = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            sb.SkipBlock(128, 192, stride = 1),
            nn.ReLU(),
            sb.SkipBlock(192, 256, stride = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256)
        )
    
    def forward(self, x):
        return self.network(x)
