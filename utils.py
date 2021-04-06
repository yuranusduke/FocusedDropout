"""
Define some utilities

Created by Kunhong Yu
Date: 2021/03/28
"""
import torch as t
from torch.nn import functional as F
from FocusedDropout import FocusedDropout

####################################
#         Define utilities         #
####################################
def _conv_layer(in_channels,
                out_channels):
    """Define conv layer
    Args :
        --in_channels: input channels
        --out_channels: output channels
    return :
        --conv layer
    """
    conv_layer = t.nn.Sequential(
        t.nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                    kernel_size = 3, stride = 1, padding = 1),
        t.nn.BatchNorm2d(out_channels),
        t.nn.ReLU(inplace = True)
    )

    return conv_layer


def vgg_block(in_channels,
              out_channels,
              repeat):
    """Define VGG block
    Args :
        --in_channels: input channels
        --out_channels: output channels
        --repeat
    return :
        --block
    """
    block = [
        _conv_layer(in_channels = in_channels if i == 0 else out_channels,
                    out_channels = out_channels)
        for i in range(repeat)
    ]

    return block

####################################
#          Define VGG16            #
####################################
class VGG16(t.nn.Module):
    """Define VGG16-style model"""

    def __init__(self, use_dropout = False, use_focused_dropout = True, dropout = 0.5, participation_rate = 0.1):
        """
        Args :
            --use_dropout: default is False
            --use_focused_dropout: default is True
            --device: learning device, default is 'cuda'
            --dropout: dropout rate, default is 0.5
            --participation_rate: default is 0.1
        """
        super(VGG16, self).__init__()

        assert use_dropout + use_focused_dropout != 2 # can't make dropout and focused dropout happen at the same time

        self.use_dropout = use_dropout
        self.use_focused_dropout = use_focused_dropout
        self.par_rate = participation_rate

        self.layer1 = t.nn.Sequential(*vgg_block(in_channels = 3,
                                                 out_channels = 64,
                                                 repeat = 2))

        self.layer2 = t.nn.Sequential(*vgg_block(in_channels = 64,
                                                 out_channels = 128,
                                                 repeat = 2))


        self.layer3 = t.nn.Sequential(*vgg_block(in_channels = 128,
                                                 out_channels = 256,
                                                 repeat = 3))


        self.layer4 = t.nn.Sequential(*vgg_block(in_channels = 256,
                                                 out_channels = 512,
                                                 repeat = 3))

        self.fc = t.nn.Sequential(  # unlike original VGG16, I reduce some fc
            # parameters to fit my 2070 device
            t.nn.Linear(512, 256),
            t.nn.ReLU(inplace = True),
            t.nn.Linear(256, 10)
        )

        self.max_pool = t.nn.MaxPool2d(kernel_size = 2, stride = 2)

        if self.use_dropout:
            self.do = t.nn.Dropout(dropout)
        if self.use_focused_dropout:
            self.fdo = FocusedDropout() # we do not modify low and high parameters here for simple verfication

    def forward(self, x):
        x1 = self.layer1(x)
        if self.use_dropout:
            x1 = self.do(x1)
        if self.use_focused_dropout:
            x1 = self.fdo(x1, self.par_rate)
        x1 = self.max_pool(x1)

        x2 = self.layer2(x1)
        if self.use_dropout:
            x2 = self.do(x2)
        if self.use_focused_dropout:
            x2 = self.fdo(x2, self.par_rate)
        x2 = self.max_pool(x2)

        x3 = self.layer3(x2)
        if self.use_dropout:
            x3 = self.do(x3)
        if self.use_focused_dropout:
            x3 = self.fdo(x3, self.par_rate)
        x3 = self.max_pool(x3)

        x4 = self.layer4(x3)
        if self.use_dropout:
            x4 = self.do(x4)
        if self.use_focused_dropout:
            x4 = self.fdo(x4, self.par_rate)
        x4 = self.max_pool(x4)

        x = F.adaptive_avg_pool2d(x4, (1, 1))
        x = x.squeeze()
        x = self.fc(x)

        return x