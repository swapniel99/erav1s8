import torch.nn as nn
import torchinfo


class ConvLayer(nn.Module):
    def __init__(self, input_c, output_c, padding=1, bias=False, skip=False, norm_type='batch', n_groups=4, dropout=0):
        super(ConvLayer, self).__init__()

        # Member Variables
        self.skip = skip
        self.norm_type = norm_type
        self.n_groups = n_groups

        self.convlayer = nn.Conv2d(input_c, output_c, 3, padding=padding, bias=bias, padding_mode='replicate')
        self.normlayer = self.get_norm_layer(output_c)
        self.actlayer = nn.ReLU()
        self.droplayer = None
        if dropout > 0:
            self.droplayer = nn.Dropout(dropout)

    def get_norm_layer(self, c):
        if self.norm_type == 'batch':
            return nn.BatchNorm2d(c)
        elif self.norm_type == 'layer':
            return nn.GroupNorm(1, c)
        elif self.norm_type == 'group':
            return nn.GroupNorm(self.n_groups, c)
        elif self.norm_type == 'instance':
            return nn.GroupNorm(c, c)

    def forward(self, x):
        x_ = x
        x = self.convlayer(x)
        x = self.normlayer(x)
        if self.skip:
            x += x_
        x = self.actlayer(x)
        if self.droplayer is not None:
            x = self.droplayer(x)
        return x


class Model(nn.Module):
    def __init__(self, norm_type='batch', n_groups=4, dropout=0, skip=False):
        super(Model, self).__init__()

        # Member Variables
        self.norm_type = norm_type
        self.n_groups = n_groups
        self.dropout = dropout

        self.cblock1 = self.get_conv_block(3, 16, reps=2, padding=0, skip=False)
        self.tblock1 = self.get_trans_block(16, 24)
        self.cblock2 = self.get_conv_block(24, 24, reps=3, padding=1, skip=skip)
        self.tblock2 = self.get_trans_block(24, 32)
        self.cblock3 = self.get_conv_block(32, 32, reps=3, padding=1, skip=skip)

        self.oblock = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 10, 1, bias=True),
            nn.Flatten(),
            nn.LogSoftmax(-1)
        )

    def get_conv_block(self, input_c, output_c, reps=1, padding=1, bias=False, skip=False):
        block = list()
        for i in range(0, reps):
            block.append(
                ConvLayer(output_c if i > 0 else input_c, output_c, padding=padding, bias=bias, skip=skip,
                          norm_type=self.norm_type, n_groups=self.n_groups, dropout=self.dropout)
            )
        return nn.Sequential(*block)

    @staticmethod
    def get_trans_block(input_c, output_c):
        return nn.Sequential(
            nn.Conv2d(input_c, output_c, 1, bias=False),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        x = self.cblock1(x)
        x = self.tblock1(x)
        x = self.cblock2(x)
        x = self.tblock2(x)
        x = self.cblock3(x)
        x = self.oblock(x)
        return x

    def summary(self, input_size=None):
        return torchinfo.summary(self, input_size=input_size,
                                 col_names=["input_size", "output_size", "num_params", "params_percent"])
