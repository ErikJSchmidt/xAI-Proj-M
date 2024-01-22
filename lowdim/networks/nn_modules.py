from torch import nn
import torch.nn.functional as F


class SkipBlock(nn.Module):
    '''
    A Res-Net-style skip-connection.
    '''

    def __init__(self, in_channels, out_channels, stride=1):
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
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = None

        # We could consider skipping other blocks
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
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
        self.in_channels = block[0].in_channels  # Find the in_channels from block, and then out_channels
        self.out_channels = block[-1].out_channels  # This is definitely going to break when you use this class wrong...
        self.stride = block[-1].stride

        self.skip = nn.Sequential()

        print("Set up Modular Skip")
        print(f"in: {self.in_channels}, out: {self.out_channels}, stride: {self.stride}")

        if self.in_channels != self.out_channels or self.stride[0] != 1 or self.stride[1] != 1:
            print("Create skip conv")
            self.skip = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=self.stride, padding=1, bias=False),
                nn.BatchNorm2d(self.out_channels)
            )
        else:
            print("No skip conv needed, use identity")
            self.skip = None

    def forward(self, x):
        id = x
        out = self.block(x)

        if self.skip is not None:
            id = self.skip(x)

        out += id
        out = F.relu(out)
        return out


class PrintLayer(nn.Module):
    def __init__(self, msg):
        super(PrintLayer, self).__init__()
        self.msg = msg

    def forward(self, x):
        # Do your print / debug stuff here
        print(self.msg)
        print(x.shape)
        return x
