import torch.nn as nn

from models.common import weightnorm_conv as conv

class edsr_res_block(nn.Module):
    """
    Residual block used in EDSR

    Args:
    in_shape        - number of input channels
    hidden_units    - number of filters in the convolutional layer
    out_shape       - number of output channels
    scaling         - Residual block scaling (sometimes used to make model stable)
    """
    def __init__(self, in_shape:int, hidden_units:int, out_shape:int, scaling:int = 1):
        super().__init__()
        self.scaling = scaling
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_shape, out_channels=hidden_units, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_units, out_channels=out_shape, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        return x + self.scaling * self.conv_block(x)

class wdsr_a_res_block(nn.Module):
    """
    'Type a' residual block used in WDSR

    Args:
    in_shape       - number of input channels
    expansion      - expansion factor for expanding the number of channels
    out_shape      - number of output channels
    scaling         - Residual block scaling (sometimes used to make model stable)
    """
    def __init__(self, in_shape:int, expansion:float, out_shape:int, scaling:float):
        super().__init__()
        self.scaling = scaling
        self.conv_block = nn.Sequential(
            conv(in_channels=in_shape, out_channels=int(expansion * in_shape), kernel_size=3),
            nn.ReLU(inplace=True),
            conv(in_channels=int(expansion * in_shape), out_channels=out_shape, kernel_size=3)
        )
    
    def forward(self, x):
        return x + self.scaling * self.conv_block(x)

class wdsr_b_res_block(nn.Module):
    """
    'Type b' residual block used in WDSR

    Args:
    in_shape    - number of input channels
    expansion   - expansion factor for expanding the number of channels
    out_shape   - number of output channels
    scaling     - Residual block scaling (sometimes used to make model stable)
    """
    def __init__(self, in_shape:int, expansion:float, out_shape:int, scaling:float):
        super().__init__()
        self.scaling = scaling
        self.conv_block = nn.Sequential(
            conv(in_channels=in_shape, out_channels=int(expansion * in_shape), kernel_size=1),
            nn.ReLU(inplace=True),
            conv(in_channels=int(expansion * in_shape), out_channels=out_shape, kernel_size=3)
        )
    
    def forward(self, x):
        return x + self.scaling * self.conv_block(x)
