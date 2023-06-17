import torch.nn as nn
import torchinfo


class Model(nn.Module):
    def __init__(self, norm_type='batch', n_groups=4):
        super(Model, self).__init__()
        self.cblock1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, bias=False),
            self.get_norm_layer(norm_type, 16, n_groups),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, bias=False),
            self.get_norm_layer(norm_type, 16, n_groups),
            nn.ReLU()
        )

        self.tblock1 = self.get_trans_block(16, 16)
        self.cblock2 = self.get_conv_block(16, 24, norm_type, n_groups)
        self.tblock2 = self.get_trans_block(24, 24)
        self.cblock3 = self.get_conv_block(24, 32, norm_type, n_groups)

        self.oblock = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 10, 1),
            nn.Flatten(),
            nn.LogSoftmax(-1)
        )

    @classmethod
    def get_conv_block(cls, input_c, output_c, norm_type, n_groups, reps=3):
        block = [
            nn.Conv2d(input_c, output_c, 3, padding=1, bias=False, padding_mode='replicate'),
            cls.get_norm_layer(norm_type, output_c, n_groups),
            nn.ReLU()
        ]
        for i in range(1, reps):
            block += [
                nn.Conv2d(output_c, output_c, 3, padding=1, bias=False, padding_mode='replicate'),
                cls.get_norm_layer(norm_type, output_c, n_groups),
                nn.ReLU()
            ]
        return nn.Sequential(*block)

    @classmethod
    def get_trans_block(cls, input_c, output_c):
        return nn.Sequential(
            nn.Conv2d(input_c, output_c, 1, bias=False),
            nn.MaxPool2d(2, 2)
        )

    @staticmethod
    def get_norm_layer(norm_type, x, n_groups):
        if norm_type == 'batch':
            return nn.BatchNorm2d(x)
        elif norm_type == 'layer':
            return nn.GroupNorm(1, x)
        elif norm_type == 'group':
            return nn.GroupNorm(n_groups, x)
        elif norm_type == 'instance':
            return nn.GroupNorm(x, x)

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
