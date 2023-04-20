import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None



class labelDiscriminator(nn.Module):
    def __init__(self, num_classes, ndf=64):
        super(labelDiscriminator, self).__init__()
        # self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2,padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2,padding=1)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2,padding=1)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2,padding=1)
        self.classifier = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2,padding=1)
        self.gpN1 = nn.GroupNorm(num_groups=32, num_channels=num_classes, eps=1e-5, affine=False)
        self.gpN2 = nn.GroupNorm(num_groups=32, num_channels=num_classes, eps=1e-5, affine=False)
        self.gpN3 = nn.GroupNorm(num_groups=32, num_channels=num_classes, eps=1e-5, affine=False)
        self.gpN4 = nn.GroupNorm(num_groups=32, num_channels=num_classes, eps=1e-5, affine=False)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gpN1(x)
        x = self.leaky_relu(x)

        # x = F.max_pool2d(x, kernel_size=2)

        x = self.conv2(x)
        x = self.gpN2(x)
        x = self.leaky_relu(x)

        # x = F.max_pool2d(x, kernel_size=2)

        x = self.conv3(x)
        x = self.gpN3(x)
        x = self.leaky_relu(x)

        # x = F.max_pool2d(x, kernel_size=2)

        x = self.conv4(x)
        x = self.gpN4(x)
        x = self.leaky_relu(x)

        # x = F.max_pool2d(x, kernel_size=2)

        x = self.classifier(x)

        # x = F.max_pool2d(x, kernel_size=2)

        return x


class featureDiscriminator(nn.Module):
    def __init__(self, input_channels, input_size, fc_classifier=3):
        super(featureDiscriminator, self).__init__()
        self.fc_classifier = fc_classifier  # 全连接层 3 layers
        self.fc_channels = [288, 144, 2]
        # self.fc_channels = [512, 256, 2]
        self.conv_channels = [16, 8, 8]

        # self.conv_channels = [input_channels//2,input_channels//4]
        self.input_size = input_size

        self.conv_features = nn.Sequential()
        self.fc_features = nn.Sequential()

        # convolutional layers
        in_channels = input_channels
        data_size = input_size
        for i, out_channels in enumerate(self.conv_channels):
            conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                 nn.GroupNorm(num_groups=4, num_channels=out_channels, eps=0, affine=False),
                                 nn.ReLU())
            self.conv_features.add_module('conv%d' % (i + 1), conv)
            in_channels = out_channels
            data_size = data_size // 2

        # full connections
        in_channels = self.conv_channels[-1] * data_size * data_size
        for i, out_channels in enumerate(self.fc_channels):
            if i == fc_classifier - 1:
                fc = nn.Sequential(nn.Linear(int(in_channels), out_channels))
            else:
                fc = nn.Sequential(nn.Linear(int(in_channels), out_channels),
                                   nn.GroupNorm(num_groups=4, num_channels=out_channels, eps=0, affine=False),
                                   nn.ReLU())
            self.fc_features.add_module('linear%d' % (i + 1), fc)
            in_channels = out_channels

    def forward(self, x):

        for i in range(len(self.conv_channels)):
            x = getattr(self.conv_features, 'conv%d' % (i + 1))(x)
            x = F.max_pool2d(x, kernel_size=2)

        x = x.view(x.size(0), -1)
        for i in range(self.fc_classifier):
            x = getattr(self.fc_features, 'linear%d' % (i + 1))(x)

        return x


class featureDiscriminator_DANN(nn.Module):
    def __init__(self, input_channels, input_size, fc_classifier=3):
        super(featureDiscriminator_DANN, self).__init__()
        self.fc_classifier = fc_classifier  # 全连接层 3 layers
        self.fc_channels = [288, 144, 2]
        # self.fc_channels = [512, 256, 2]
        self.conv_channels = [16, 8, 8]
        # self.conv_channels = [input_channels//2,input_channels//4]
        self.input_size = input_size

        self.conv_features = nn.Sequential()
        self.fc_features = nn.Sequential()

        # convolutional layers
        in_channels = input_channels
        data_size = input_size
        for i, out_channels in enumerate(self.conv_channels):
            conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                 nn.GroupNorm(num_groups=4, num_channels=out_channels, eps=0, affine=False),
                                 nn.ReLU())
            self.conv_features.add_module('conv%d' % (i + 1), conv)
            in_channels = out_channels
            data_size = data_size // 2

        # full connections
        in_channels = self.conv_channels[-1] * data_size * data_size
        for i, out_channels in enumerate(self.fc_channels):
            if i == fc_classifier - 1:
                fc = nn.Sequential(nn.Linear(int(in_channels), out_channels))
            else:
                fc = nn.Sequential(nn.Linear(int(in_channels), out_channels),
                                   nn.GroupNorm(num_groups=4, num_channels=out_channels, eps=0, affine=False),
                                   nn.ReLU())
            self.fc_features.add_module('linear%d' % (i + 1), fc)
            in_channels = out_channels

    def forward(self, x,alpha):

        for i in range(len(self.conv_channels)):
            x = getattr(self.conv_features, 'conv%d' % (i + 1))(x)
            x = F.max_pool2d(x, kernel_size=2)
        x = x.view(x.size(0), -1)
        reverse_x = ReverseLayerF.apply(x, alpha)
        for i in range(self.fc_classifier):
            reverse_x = getattr(self.fc_features, 'linear%d' % (i + 1))(reverse_x)
        return reverse_x

class featureDiscriminator_DANN2(nn.Module):
    def __init__(self, input_channels, input_size, fc_classifier=3):
        super(featureDiscriminator_DANN2, self).__init__()
        self.fc_classifier = fc_classifier  # 全连接层 3 layers
        self.fc_channels = [288, 144, 2]
        # self.fc_channels = [512, 256, 2]
        self.conv_channels = [16, 16, 16]
        # self.conv_channels = [input_channels//2,input_channels//4]
        self.input_size = input_size

        self.conv_features = nn.Sequential()
        self.fc_features = nn.Sequential()

        # convolutional layers
        in_channels = input_channels
        data_size = input_size
        for i, out_channels in enumerate(self.conv_channels):
            conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                 nn.GroupNorm(num_groups=4, num_channels=out_channels, eps=0, affine=False),
                                 nn.ReLU())
            self.conv_features.add_module('conv%d' % (i + 1), conv)
            in_channels = out_channels
            data_size = data_size // 2

        # full connections
        in_channels = self.conv_channels[-1] * data_size * data_size
        for i, out_channels in enumerate(self.fc_channels):
            if i == fc_classifier - 1:
                fc = nn.Sequential(nn.Linear(int(in_channels), out_channels))
            else:
                fc = nn.Sequential(nn.Linear(int(in_channels), out_channels),
                                   nn.GroupNorm(num_groups=4, num_channels=out_channels, eps=0, affine=False),
                                   nn.ReLU())
            self.fc_features.add_module('linear%d' % (i + 1), fc)
            in_channels = out_channels

    def forward(self, x,alpha):

        for i in range(len(self.conv_channels)):
            x = getattr(self.conv_features, 'conv%d' % (i + 1))(x)
            x = F.max_pool2d(x, kernel_size=2)
        x = x.view(x.size(0), -1)
        reverse_x = ReverseLayerF.apply(x, alpha)
        for i in range(self.fc_classifier):
            reverse_x = getattr(self.fc_features, 'linear%d' % (i + 1))(reverse_x)
        return reverse_x

class featureDiscriminator_DANN3(nn.Module):  #revise conv layers
    def __init__(self, input_channels, input_size, fc_classifier=3):
        super(featureDiscriminator_DANN3, self).__init__()
        self.fc_classifier = fc_classifier  # 全连接层 3 layers
        self.fc_channels = [288, 144, 2]
        # self.fc_channels = [512, 256, 2]
        self.conv_channels = [16, 16, 16]
        # self.conv_channels = [input_channels//2,input_channels//4]
        self.input_size = input_size

        self.conv_features = nn.Sequential()
        self.fc_features = nn.Sequential()

        # convolutional layers
        in_channels = input_channels
        data_size = input_size
        for i, out_channels in enumerate(self.conv_channels):
            conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                 nn.GroupNorm(num_groups=4, num_channels=out_channels, eps=0, affine=False),
                                 nn.ReLU())
            self.conv_features.add_module('conv%d' % (i + 1), conv)
            in_channels = out_channels
            data_size = data_size // 2

        # full connections
        in_channels = self.conv_channels[-1] * data_size * data_size
        for i, out_channels in enumerate(self.fc_channels):
            if i == fc_classifier - 1:
                fc = nn.Sequential(nn.Linear(int(in_channels), out_channels))
            else:
                fc = nn.Sequential(nn.Linear(int(in_channels), out_channels),
                                   nn.GroupNorm(num_groups=4, num_channels=out_channels, eps=0, affine=False),
                                   nn.ReLU())
            self.fc_features.add_module('linear%d' % (i + 1), fc)
            in_channels = out_channels

    def forward(self, x,alpha):
        x = ReverseLayerF.apply(x, alpha)
        for i in range(len(self.conv_channels)):
            x = getattr(self.conv_features, 'conv%d' % (i + 1))(x)
            x = F.max_pool2d(x, kernel_size=2)
        x = x.view(x.size(0), -1)

        for i in range(self.fc_classifier):
            x = getattr(self.fc_features, 'linear%d' % (i + 1))(x)
        return x




if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    model_feature = featureDiscriminator_DANN(input_channels=16, input_size=544, fc_classifier=3).cuda()
    img = torch.randn((2,16,544,544)).cuda()
    out = model_feature(img,alpha=1.5)
    print(out)
    macs, params = get_model_complexity_info(model_feature(), (16,544,544), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))