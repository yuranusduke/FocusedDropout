"""
Simple implementation of Focused Dropout
From paper : <FocusedDropout for Convolutional Neural Network>

Created by Kunhong Yu
Date : 2021/03/31
"""
import torch as t

class FocusedDropout(t.nn.Module):
    """Define Focused Dropout module"""

    def __init__(self, low = 0.6, high = 0.9):
        """
        Args :
            --low: left value in random range, default is 0.6
            --high: right value in random range, default is 0.9
        """
        super(FocusedDropout, self).__init__()

        self.low = low
        self.high = high
        self.avg_pool = t.nn.AdaptiveAvgPool2d(1)

    def forward(self, x, par_rate):
        if self.train():
            # 1. First, we need to do global average pooling
            x_ = self.avg_pool(x) # [m, C, 1, 1]
            x_ = x_.squeeze() # [m, C]

            # 2. Find the maximum one
            x_max_val, _ = t.max(x_, dim = -1, keepdim = True) # [m, 1]
            hidden_mask = (x_ == x_max_val) # [m, C]
            hidden_mask = t.unsqueeze(t.unsqueeze(hidden_mask, dim = -1), dim = -1)
            x_max = x * hidden_mask
            x_max = t.max(t.max(t.sum(x_max, dim = 1), dim = -1)[0], dim = -1)[0] # [m, 1]
            x_max = t.unsqueeze(x_max, dim = -1)

            #x_max, _ = t.max(x[x_max_indice, ...], dim = -1, keepdim = True) # [m, 1]
            x_max = t.unsqueeze(t.unsqueeze(x_max, dim = -1), dim = -1) # [m, 1, 1, 1]
            x_max = x_max.repeat(1, x.size(1), x.size(2), x.size(3))
            mask = t.zeros_like(x) # [m, C, H, W]

            # 3. sample
            rand = t.rand_like(x)
            rand = rand * (self.high - self.low) + self.low # [0, 1] -> [low, high]

            x_max *= rand # [m, C, H, W]
            indices = x > x_max
            mask += indices

            examples = x.size(0)
            num_par = int(par_rate * examples)
            mask[num_par:] = 1.

            # 4. Focused Dropout on some content

            return x * mask

        else:
            return x