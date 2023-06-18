import torch.nn as nn
import torchinfo


class Model(nn.Module):
    def __init__(self, norm_type='batch', n_groups=4):
        super(Model, self).__init__()

        # Member Variables
        self.norm_type = norm_type
        self.n_groups = n_groups

        self.cblock1 = self.get_conv_block(3, 16, padding=0, reps=2)
        self.tblock1 = self.get_trans_block(16, 16)
        self.cblock2 = self.get_conv_block(16, 24)
        self.tblock2 = self.get_trans_block(24, 24)
        self.cblock3 = self.get_conv_block(24, 32)

        self.oblock = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 10, 1),
            nn.Flatten(),
            nn.LogSoftmax(-1)
        )

    def get_conv_block(self, input_c, output_c, padding=1, bias=False, reps=3, padding_mode='replicate'):
        block = [
            nn.Conv2d(input_c, output_c, 3, padding=padding, bias=bias, padding_mode=padding_mode),
            self.get_norm_layer(output_c),
            nn.ReLU()
        ]
        for i in range(1, reps):
            block += [
                nn.Conv2d(output_c, output_c, 3, padding=padding, bias=bias, padding_mode=padding_mode),
                self.get_norm_layer(output_c),
                nn.ReLU()
            ]
        return nn.Sequential(*block)

    @staticmethod
    def get_trans_block(input_c, output_c):
        return nn.Sequential(
            nn.Conv2d(input_c, output_c, 1, bias=False),
            nn.MaxPool2d(2, 2)
        )

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
