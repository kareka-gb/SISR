import torch
import torch.nn as nn

from torch.nn.utils import weight_norm

def weightnorm_conv(in_channels, out_channels, kernel_size, **kwargs):
    """
    Returns a weight normalized convolutional layer
    """
    return weight_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size//2, **kwargs))

class upsample(nn.Module):
    """
    Upsampling block

    Args:
    in_shape - number of input channels
    out_shape - number of output channels
    scale - Scale by which the image should be upsampled
    bn - Whether to use batch normalization or not
    act - Whether to use ReLU activation or not
    model - for which model upsampling is used
    """
    def __init__(self, shape:int, scale:int, bn:bool = False, act:bool = False, model:str = 'edsr', kernel_size:int = 3):
        super().__init__()
        self.bn = bn
        self.act = act
        self.conv1 = nn.Conv2d(in_channels=shape, out_channels=shape * scale * scale, kernel_size=kernel_size, padding=kernel_size//2)
        if model == 'wdsr':
            self.conv1 = weight_norm(self.conv1)
        elif model != 'edsr':
            NotImplementedError()
        self.shuffle = nn.PixelShuffle(scale)
        self.batch_norm = nn.BatchNorm2d(shape)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.shuffle(self.conv1(x))
        if self.bn:
            x = self.batch_norm(x)
        if self.act:
            return self.relu(x)
        return x

class MeanShift(nn.Conv2d):
    """
    Used to include mean shift in the model

    Args:
    rgb_mean - mean of RGB images of dataset
    rgb_std - std of RGB images of dataset (use if required)
    sign - to use inverse mean shift

    Div2k mean - (0.4458, 0.4367, 0.4043)
    Div2k std - (0.2841, 0.2702, 0.2924)
    """
    def __init__(self, shape: int = 3, rgb_range: int = 1.0, rgb_mean=(0.4468, 0.4367, 0.4043), rgb_std=(1.0, 1.0, 1.0), sign: int = -1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

