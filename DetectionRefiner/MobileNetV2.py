import torch
import torch.nn as nn

# Taken from https://github.com/ericsun99/MobileNet-V2-Pytorch/blob/master/MobileNetV2.py

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*self.features)

        self.linear1 = nn.Linear(1280, 128)
        self.linear2 = nn.Linear(128, 4)
        self.linear3 = nn.Linear(1284, 128)
        self.linear4 = nn.Linear(128, 4)
        self.relu = nn.ReLU6(inplace=True)
        self.sigmoid = nn.Sigmoid()


    def forward(self, image, unrefined_bounding_box):
        x = self.features(image)
        # features = x.view(-1, self.last_channel)
        features = x.view(x.size(0), -1)

        # First regression head, directly estimate the bounding box
        head_1 = self.relu(self.linear1(features))
        directly_predicted_bounding_box = self.sigmoid(self.linear2(head_1))

        # Second regression head, merge in the unrefined bounding box
        head_2 = torch.cat((features, unrefined_bounding_box), dim=1)
        head_2 = self.relu(self.linear3(head_2))
        guided_predicted_bounding_box = self.sigmoid(self.linear4(head_2))

        return directly_predicted_bounding_box, guided_predicted_bounding_box