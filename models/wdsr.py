from torch import nn

from models.blocks import wdsr_a_res_block, wdsr_b_res_block
from models.common import upsample, MeanShift
from models.common import weightnorm_conv as conv

def res_block(in_shape, expansion, out_shape, scaling, type:str='b'):
    if type == 'a':
        return wdsr_a_res_block(in_shape, expansion, out_shape, scaling)
    if type == 'b':
        return wdsr_b_res_block(in_shape, expansion, out_shape, scaling)
    NotImplementedError()

class wdsr(nn.Module):
    """
    WDSR Neural Network

    Args:
    type            - Type of residual block to use 'a' or 'b'
    shape           - number of input channels: default = 3
    hidden_units    - number of filters in the middle of the network: default = 64
    num_res_blocks  - number of residual blocks in the network: default = 8
    scale           - upscaling factor
    res_block_expansion - Expansion that should be used in residual blocks for wider activation
    res_block_scaling   - Scaling factor in residual blocks (sometimes used to make model stable)
    """
    def __init__(self, type:str = 'b', shape:int = 3, hidden_units:int = 64, num_res_blocks:int = 8, 
                 scale:int = 4, res_block_expansion:float = 2.0, res_block_scaling:float = 1.0):
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
        head = [conv(shape, hidden_units, 3)]
        body = [res_block(type=type, in_shape=hidden_units, expansion=res_block_expansion, out_shape=hidden_units, scaling=res_block_scaling) for _ in range(num_res_blocks)]
        tail = [upsample(shape=hidden_units, scale=f, model='wdsr') for f in self.scale_factors]
        tail.append(conv(hidden_units, shape, 3))
        skip = [upsample(shape=shape, scale=scale, model='wdsr', kernel_size=5)]

        self.sub_mean = MeanShift(sign=-1)
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        self.skip = nn.Sequential(*skip)
        self.add_mean = MeanShift(sign=1)
    
    def forward(self, x):
        x = self.sub_mean(x)

        b = self.skip(x)
        x = self.tail(self.body(self.head(x)))

        x = self.add_mean(x + b)
        return x