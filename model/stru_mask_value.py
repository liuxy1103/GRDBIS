from __future__ import absolute_import
from __future__ import division

__all__ = ['StRABottleneck', 'init_pretrained_weights']
import torch.nn as nn
from torchvision.models.resnet import Bottleneck, conv1x1
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


class StRABottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ratio, stride=1, downsample=None, rate=1, dilation=1, kernel_size=3,
                 base_width=64, groups=8, sa_conv=None, stage1=False):
        super(StRABottleneck, self).__init__()

        width = int(planes * (base_width / 64.))
        self.stride = stride
        self.stage1 = stage1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = sa_conv(width, out_channels=width//rate, value_out_channels=width, dilation=dilation,
                             stride=stride, groups=groups, kernel_size=kernel_size, ratio=ratio)

        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

