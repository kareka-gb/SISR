from torch import nn

from models.blocks import edsr_res_block as res_block
from models.common import upsample, MeanShift

class edsr(nn.Module):
    """
    Enhanced Deep Super Resolution Neural Network

    Args: 
    shape           - number of channels in input image
    hidden_units    - number of filters used in residual blocks of the network. Use the number in proportion to the scale
    num_res_blocks  - number of residual blocks in the network
    scale           - Scaling factor by which we want to super resolve the image
    
    """
    def __init__(self, shape:int=3, hidden_units:int=64, num_res_blocks:int=8, scale:int=4):
        super().__init__()
        self.scale_factors = []
        if scale == 2:
            self.scale_factors.append(2)
        elif scale == 3:
            self.scale_factors.append(3)
        elif scale == 4:
            self.scale_factors.append(2)
            self.scale_factors.append(2)
        elif scale == 6:
            self.scale_factors.append(2)
            self.scale_factors.append(3)
        elif scale == 8:
            self.scale_factors.append(2)
            self.scale_factors.append(2)
            self.scale_factors.append(2)
        else:
            raise Exception("Scaling factor must be one of (2, 3, 4, 6, 8)")
        self.sub_mean = MeanShift(sign=-1)
        self.conv1 = nn.Conv2d(in_channels=shape, out_channels=hidden_units, kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList([res_block(in_shape=hidden_units, hidden_units=hidden_units, out_shape=hidden_units) for i in range(num_res_blocks)])
        self.conv2 = nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1)
        self.up = nn.ModuleList([upsample(shape=hidden_units, scale=f) for f in self.scale_factors])
        self.conv3 = nn.Conv2d(in_channels=hidden_units, out_channels=shape, kernel_size=3, padding=1)
        self.add_mean = MeanShift(sign=1)
    
    def forward(self, x):
        x = self.sub_mean(x)
        x = b = self.conv1(x)
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        x = self.conv2(x + b)
        # Upsampling
        for block in self.up:
            x = block(x)
        x = self.conv3(x)
        x = self.add_mean(x)
        return x
