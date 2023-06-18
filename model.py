import torch.nn as nn
import torchinfo

from ghostbn import GhostBatchNorm


# ****************               ASSIGNMENT 8 Models                ******************** #


class ConvLayer(nn.Module):
    def __init__(self, input_c, output_c, padding=1, bias=False, skip=False, norm_type=None, n_groups=4, dropout=0):
        super(ConvLayer, self).__init__()

        # Member Variables
        self.skip = skip
        self.norm_type = norm_type
        self.n_groups = n_groups

        self.convlayer = nn.Conv2d(input_c, output_c, 3, padding=padding, bias=bias, padding_mode='replicate')
        self.normlayer = None
        if self.norm_type is not None:
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
        elif self.norm_type == 'ghost':
            return GhostBatchNorm(c, self.n_groups)
        else:
            raise Exception(f'Unknown norm_type: {self.norm_type}')

    def forward(self, x):
        x_ = x
        x = self.convlayer(x)
        if self.normlayer is not None:
            x = self.normlayer(x)
        if self.skip:
            x += x_
        x = self.actlayer(x)
        if self.droplayer is not None:
            x = self.droplayer(x)
        return x


class Model(nn.Module):
    def __init__(self, norm_type=None, n_groups=4, dropout=0, skip=False):
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


class GroupNormModel(Model):
    def __init__(self, n_groups=4, dropout=0, skip=False):
        super(GroupNormModel, self).__init__(norm_type='group', n_groups=n_groups, dropout=dropout, skip=skip)


class LayerNormModel(Model):
    def __init__(self, dropout=0, skip=False):
        super(LayerNormModel, self).__init__(norm_type='layer', dropout=dropout, skip=skip)


class BatchNormModel(Model):
    def __init__(self, dropout=0, skip=False):
        super(BatchNormModel, self).__init__(norm_type='batch', dropout=dropout, skip=skip)


class GhostBatchNormModel(Model):
    def __init__(self, dropout=0, skip=False):
        super(GhostBatchNormModel, self).__init__(norm_type='ghost', dropout=dropout, skip=skip)


# ****************               ASSIGNMENT 7 Models                ******************** #


class BaseModel(nn.Module):
    def summary(self, input_size=None):
        return torchinfo.summary(
            self,
            input_size=input_size,
            col_names=["input_size", "output_size", "num_params", "params_percent"],
        )


class Model2(BaseModel):
    def __init__(self):
        super(Model2, self).__init__()
        self.cblock1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.tblock1 = nn.Sequential(nn.MaxPool2d(2, 2))

        self.cblock2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.tblock2 = nn.Sequential(nn.MaxPool2d(2, 2))

        self.cblock3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.oblock = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 256, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(256, 10, 1),
            nn.Flatten(),
            nn.LogSoftmax(-1),
        )

    def forward(self, x):
        x = self.cblock1(x)
        x = self.tblock1(x)
        x = self.cblock2(x)
        x = self.tblock2(x)
        x = self.cblock3(x)
        x = self.oblock(x)
        return x


class Model3(BaseModel):
    def __init__(self):
        super(Model3, self).__init__()
        self.cblock1 = nn.Sequential(
            nn.Conv2d(1, 9, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(9, 10, 3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.tblock1 = nn.Sequential(nn.MaxPool2d(2, 2))

        self.cblock2 = nn.Sequential(
            nn.Conv2d(10, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.tblock2 = nn.Sequential(nn.MaxPool2d(2, 2))

        self.cblock3 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.oblock = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16, 32, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 10, 1),
            nn.Flatten(),
            nn.LogSoftmax(-1),
        )

    def forward(self, x):
        x = self.cblock1(x)
        x = self.tblock1(x)
        x = self.cblock2(x)
        x = self.tblock2(x)
        x = self.cblock3(x)
        x = self.oblock(x)
        return x


class Model4(BaseModel):
    def __init__(self):
        super(Model4, self).__init__()
        self.cblock1 = nn.Sequential(
            nn.Conv2d(1, 9, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(9),
            nn.Conv2d(9, 10, 3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.tblock1 = nn.Sequential(nn.MaxPool2d(2, 2))

        self.cblock2 = nn.Sequential(
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.tblock2 = nn.Sequential(nn.MaxPool2d(2, 2))

        self.cblock3 = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.oblock = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 10, 1),
            nn.Flatten(),
            nn.LogSoftmax(-1),
        )

    def forward(self, x):
        x = self.cblock1(x)
        x = self.tblock1(x)
        x = self.cblock2(x)
        x = self.tblock2(x)
        x = self.cblock3(x)
        x = self.oblock(x)
        return x


class Model5(BaseModel):
    def __init__(self):
        DROP = 0.01
        super(Model5, self).__init__()
        self.cblock1 = nn.Sequential(
            nn.Conv2d(1, 9, 3, padding=1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
            nn.BatchNorm2d(9),
            nn.Conv2d(9, 10, 3, padding=1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
        )

        self.tblock1 = nn.Sequential(nn.MaxPool2d(2, 2))

        self.cblock2 = nn.Sequential(
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 16, 3, padding=1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
        )

        self.tblock2 = nn.Sequential(nn.MaxPool2d(2, 2))

        self.cblock3 = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
        )

        self.oblock = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 1, bias=False),
            nn.Dropout(DROP),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 10, 1),
            nn.Flatten(),
            nn.LogSoftmax(-1),
        )

    def forward(self, x):
        x = self.cblock1(x)
        x = self.tblock1(x)
        x = self.cblock2(x)
        x = self.tblock2(x)
        x = self.cblock3(x)
        x = self.oblock(x)
        return x
