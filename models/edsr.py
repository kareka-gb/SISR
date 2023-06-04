from torch import nn

class res_block(nn.Module):
    """
    Residual block used in EDSR

    Args:
    in_shape        - number of input channels
    hidden_units    - number of filters in the convolutional layer
    out_shape       - number of output channels
    """
    def __init__(self, in_shape:int, hidden_units:int, out_shape:int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_shape, out_channels=hidden_units, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_units, out_channels=out_shape, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        return x + self.conv_block(x)


class upsample(nn.Module):
    """
    Upsampling block

    Args:
    in_shape - number of input channels
    out_shape - number of output channels
    scale - Scale by which the image should be upsampled
    """
    def __init__(self, in_shape:int, out_shape:int, scale:int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_shape, out_channels=out_shape * scale * scale, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)
        self.conv2 = nn.Conv2d(in_channels=out_shape, out_channels=out_shape, kernel_size=3, padding=1)
    
    def forward(self, x):
        return self.conv2(self.shuffle(self.conv1(x)))

class edsr(nn.Module):
    """
    Enhanced Deep Super Resolution Neural Network

    Args: 
    shape           - number of channels in input image
    hidden_units    - number of filters used in residual blocks of the network. Use the number in proportion to the scale
    num_res_blocks  - number of residual blocks in the network
    scale           - Scaling factor by which we want to super resolve the image
    
    """
    def __init__(self, shape:int, hidden_units:int=64, num_res_blocks:int=8, scale:int=4):
        super().__init__()
        self.scale_factors = []
        self.up_in_shapes = []
        self.up_out_shapes = []
        if scale == 2:
            self.scale_factors.append(2); self.up_in_shapes.append(hidden_units); self.up_out_shapes.append(shape);
        elif scale == 3:
            self.scale_factors.append(3); self.up_in_shapes.append(hidden_units); self.up_out_shapes.append(shape);
        elif scale == 4:
            self.scale_factors.append(2); self.up_in_shapes.append(hidden_units); self.up_out_shapes.append(shape*2*2);
            self.scale_factors.append(2); self.up_in_shapes.append(shape*2*2);    self.up_out_shapes.append(shape);
        elif scale == 6:
            self.scale_factors.append(2); self.up_in_shapes.append(hidden_units); self.up_out_shapes.append(shape*3*3);
            self.scale_factors.append(3); self.up_in_shapes.append(shape*3*3);    self.up_out_shapes.append(shape);
        elif scale == 8:
            self.scale_factors.append(2); self.up_in_shapes.append(hidden_units);  self.up_out_shapes.append(shape*2*2*2*2);
            self.scale_factors.append(2); self.up_in_shapes.append(shape*2*2*2*2); self.up_out_shapes.append(shape*2*2);
            self.scale_factors.append(2); self.up_in_shapes.append(shape*2*2);     self.up_out_shapes.append(shape);
        else:
            raise Exception("Scaling factor must be one of (2, 3, 4, 6, 8)")
        self.conv1 = nn.Conv2d(in_channels=shape, out_channels=hidden_units, kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList([res_block(in_shape=hidden_units, hidden_units=hidden_units, out_shape=hidden_units) for i in range(num_res_blocks)])
        self.conv2 = nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1)
        self.up = nn.ModuleList([upsample(in_shape=self.up_in_shapes[i], out_shape=self.up_out_shapes[i], scale=f) for i, f in enumerate(self.scale_factors)])
    
    def forward(self, x_in):
        x = b = self.conv1(x_in)
        for block in self.res_blocks:
            x = block(x)
        x = self.conv2(x + b)
        for block in self.up:
            x = block(x)
        return x
