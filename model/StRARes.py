from __future__ import absolute_import
from __future__ import division
__all__ = ['stru_resnet50']
import torch
import torch.nn.functional as F
# from .component import *
from component import *
from torchvision.models.resnet import Bottleneck, conv1x1
import torch.nn as nn
# from .stru_mask_value import StRABottleneck, init_pretrained_weights
from stru_mask_value import StRABottleneck, init_pretrained_weights

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}

fnorm_loss = torch.zeros(16)  # 16 samples per gpu, set to 0 every forward process and accumulate the diversity loss
fnorm_loss = fnorm_loss.cpu()


class LocalAttn(nn.Module):
    def __init__(self, in_channels, out_channels, value_out_channels, kernel_size=3, dilation=1, stride=2, groups=8,
                 bias=False, ratio=4):
        super(LocalAttn, self).__init__()
        self.out_channels = out_channels
        self.stride = stride
        self.padding = ((kernel_size - 1) * dilation + 2 - stride) // 2
        self.groups = groups
        self.kernel_size = kernel_size  # original version that heads use the same kernel size
        self.dilation = dilation  # original version that heads use the same dilation

        self.mask_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, groups=groups, kernel_size=1, padding=0, stride=stride),
            nn.BatchNorm2d( in_channels // ratio),
            nn.Tanh(),
            nn.Conv2d(in_channels // ratio, groups * (1 + kernel_size * kernel_size), groups=groups, kernel_size=1, padding=0),
            nn.BatchNorm2d(groups * (1 + kernel_size * kernel_size))
        )
        self.value_conv = nn.Conv2d(in_channels, value_out_channels, kernel_size=1, bias=bias, groups=self.groups, padding=0)

        # (2p-(kernel-1)d-1)/stride+1=0
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, dilation=dilation, padding=self.padding)
        self.moe = MoE(groups=groups)
        self.reset_parameters()

    def forward(self, x):
        b, c, h, w = x.detach().size() # [4, 512, 34, 34]
        h = h // self.stride # 34
        w = w // self.stride # 34

        mask_neighbor = self.mask_conv(x)  # b,g*(1+k*k),h,w  # [4, 80, 34, 34]
        neighbor_feature = mask_neighbor[:, :self.groups, :, :]  # b,g,h,w  # [4, 8, 34, 34]
        mask = mask_neighbor[:, self.groups:, :, :]  # b,g*k*k,h,w  # [4, 72, 34, 34]
        neighbor_feature = self.unfold(neighbor_feature)  # b,g*k*k,h,w  # [4, 72, 1156]
        neighbor_feature = neighbor_feature.view(b, self.groups, self.kernel_size * self.kernel_size, h, w)  # b,g,k*k,h,w  # [4, 8, 9, 34, 34]
        mask = mask.view(b, self.groups, self.kernel_size * self.kernel_size, h, w)  # b,g,k*k,h,w  # [4, 8, 9, 34, 34]
        mask = mask + neighbor_feature  # b,g,k*k,h,w  # [4, 8, 9, 34, 34]
        mask = F.softmax(mask, dim=2)  # [4, 8, 9, 34, 34]
        mask = mask.unsqueeze(2)   # b,g,1,k*k,h,w  # [4, 8, 1, 9, 34, 34]

        value = self.value_conv(x)  # [4, 512, 34, 34]
        value = self.unfold(value)  # b,c*k*k,h,w  # [4, 4608, 1156]
        value = value.view(b, self.groups, self.out_channels // self.groups, -1, h, w)  # b,g,out/g,k*k,h,w  # [4, 8, 64, 9, 34, 34]

        out = mask * value  # b,g,out/g,k*k,h,w  # [4, 8, 64, 9, 34, 34]
        out = torch.sum(out, dim=3, keepdim=False)  # b,g,out/g,h,w  # [4, 8, 64, 34, 34]
        out = out.view(b, -1, h, w)  # [4, 512, 34, 34]
        out, fnorm = self.moe(out)  # [4, 512, 34, 34], [4]
        global fnorm_loss
        fnorm = F.pad(input=fnorm, pad=(16-fnorm.size(0), 0))  # to avoid error when testing  # [16]
        fnorm = fnorm.cpu()
        fnorm_loss = fnorm_loss + fnorm  # [16]
        return out  # [4, 512, 34, 34]

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class StRAResNet(nn.Module):
    def __init__(self, layers, loss, num_classes, ratio,
                 sa_dilation=1, norm_layer=None, width_per_group=64, head=8, rate=1,
                 kernel_size=3, last_stride=1, sa_conv=None, **kwargs):
        super(StRAResNet, self).__init__()
        print('kernel sizes: {}, dilation: {}, heads:{}.'.format(str(kernel_size), str(sa_dilation), head))
        self.inplanes = 64
        self.head_count = head
        self.loss = loss
        self.dilation = 1
        self.ratio = ratio
        self.sa_dilation = sa_dilation
        self.groups = 1
        self.base_width = width_per_group
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(Bottleneck, 64, layers[0])
        self.layer2 = self._make_layer(Bottleneck, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, layers[2], stride=2)
        self.layer4 = self._make_layer_sa(StRABottleneck, 512, layers[3], stride=last_stride, rate=rate,
                                          kernel_size=kernel_size, sa_conv=sa_conv)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(2048, 512, kernel_size=1, stride=1, groups=1, bias=False), nn.BatchNorm2d(512))
        self.classifier = nn.Linear(512, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _make_layer_sa(self, block, planes, blocks, stride, rate, kernel_size, sa_conv):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, groups=self.head_count,
                            rate=rate, dilation=self.sa_dilation, ratio=self.ratio, sa_conv=sa_conv, stage1=True))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.head_count, rate=rate, dilation=self.sa_dilation,
                                kernel_size=kernel_size, ratio=self.ratio, sa_conv=sa_conv))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        v = self.global_avg_pool(x)
        v = v.view(v.size(0), -1)
        global fnorm_loss
        fnorm = fnorm_loss  # restore a copy of diversity loss for backward
        fnorm_loss = torch.zeros(16)  # reset to 0 at the end of forward process
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y, fnorm.cuda()
        elif self.loss == 'triplet':
            return y, v


def stru_resnet50(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = StRAResNet(
        layers=[3, 4, 6, 3],
        loss=loss,
        head=8,
        rate=1,
        sa_dilation=1,
        last_stride=1,
        kernel_size=3,
        ratio=4,
        sa_conv=LocalAttn,
        num_classes=num_classes
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model


if __name__ == "__main__":
    import numpy as np

    x = torch.Tensor(np.random.random((4, 3, 544, 544)).astype(np.float32)).cuda()
    model = stru_resnet50(num_classes=2).cuda()

    pred, fnorm = model(x)
    print(pred.shape, fnorm.shape)