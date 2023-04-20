import torch
import torch.nn as nn
import torch.nn.functional as F

class Modified3DUNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, base_n_filter=64):
        super(Modified3DUNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.6)
        self.upsacle = nn.Upsample(scale_factor=2, mode='trilinear',align_corners=False)
        self.softmax = nn.Softmax(dim=1)

        # self.pooling1 = torch.nn.MaxPool3d(kernel_size=(1, 2, 2))
        #
        # self.full2 = self.conv_norm_lrelu(1, self.base_n_filter * 2)
        # self.full3 = self.conv_norm_lrelu(1, self.base_n_filter * 4)
        # self.full4 = self.conv_norm_lrelu(1, self.base_n_filter * 8)
        # self.full5 = self.conv_norm_lrelu(1, self.base_n_filter * 16)
        N = 64
        # stem net
        self.conv1 = nn.Conv3d(1, N, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.IN1 = nn.InstanceNorm3d(N )
        self.conv2 = nn.Conv3d(N, N, kernel_size=3, stride=(1, 2, 2), padding=1,
                               bias=False)
        self.IN2 = nn.InstanceNorm3d(N )
        self.relu = nn.LeakyReLU(inplace=False)

        # Level 1 context pathway
        self.conv3d_c1_1 = nn.Conv3d(N, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1,
                                     bias=False)
        self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
        self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)

        # Level 2 context pathway
        self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter * 2, kernel_size=3, stride=(1, 2, 2),
                                   padding=(1, 1, 1), bias=False)
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter * 2, self.base_n_filter * 2)
        self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter * 2)

        # Level 3 context pathway
        self.conv3d_c3 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 4, kernel_size=3, stride=(1, 2, 2),
                                   padding=(1, 1, 1), bias=False)
        self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter * 4, self.base_n_filter * 4)
        self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter * 4)

        # Level 4 context pathway
        self.conv3d_c4 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 8, kernel_size=3, stride=(1, 2, 2),
                                   padding=(1, 1, 1), bias=False)
        self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter * 8, self.base_n_filter * 8)
        self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter * 8)

        # Level 5 context pathway, level 0 localization pathway
        self.conv3d_c5 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 16, kernel_size=3, stride=(1, 2, 2),
                                   padding=(1, 1, 1), bias=False)
        self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter * 16, self.base_n_filter * 16)
        self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 16,
                                                                                             self.base_n_filter * 8)
        self.upl0 = self.up(self.base_n_filter * 16, self.base_n_filter * 8)

        self.conv3d_l0 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter * 8)

        # Level 1 localization pathway
        self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter * 16, self.base_n_filter * 16)
        self.conv3d_l1 = nn.Conv3d(self.base_n_filter * 16, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l1 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 8,
                                                                                             self.base_n_filter * 4)
        self.upl1 = self.up(self.base_n_filter * 8, self.base_n_filter * 4)

        # Level 2 localization pathway
        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter * 8, self.base_n_filter * 8)
        self.conv3d_l2 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 4, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 4,
                                                                                             self.base_n_filter * 2)
        self.upl2 = self.up(self.base_n_filter * 4, self.base_n_filter * 2)

        # Level 3 localization pathway
        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter * 4, self.base_n_filter * 4)
        self.conv3d_l3 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 2, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 2,
                                                                                             self.base_n_filter)
        self.upl3 = self.up(self.base_n_filter * 2, self.base_n_filter)

        # Level 4 localization pathway
        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter * 2, self.base_n_filter * 2)
        self.conv3d_l4 = nn.Conv3d(self.base_n_filter * 2, self.n_classes, kernel_size=1, stride=1, padding=0,
                                   bias=False)

        self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter * 8, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=False)
        self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter * 4, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=False)

    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU())

    def up(self, feat_in, feat_out):
        return nn.Sequential(
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def forward(self, x):
        # mult-level inputs
        # x2 = self.pooling1(x)
        # x3 = self.pooling1(x2)
        # x4 = self.pooling1(x3)
        # x5 = self.pooling1(x4)
        # x2 = self.full2(x2)
        # x3 = self.full3(x3)
        # x4 = self.full4(x4)
        # x5 = self.full5(x5)
        # stem layer
        x = self.conv1(x)
        x = self.IN1(x)
        x = self.conv2(x)
        x = self.IN2(x)
        x = self.relu(x)

        #  Level 1 context pathway
        out = self.conv3d_c1_1(x)
        residual_1 = out
        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)
        # Element Wise Summation
        out += residual_1
        context_1 = self.lrelu(out)
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)

        # Level 2 context pathway

        out = self.conv3d_c2(out)
        # out = x2 + out
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm3d_c2(out)
        out = self.lrelu(out)
        context_2 = out

        # Level 3 context pathway

        out = self.conv3d_c3(out)
        # out = x3 + out
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c3(out)
        out += residual_3
        out = self.inorm3d_c3(out)
        out = self.lrelu(out)
        context_3 = out

        # Level 4 context pathway

        out = self.conv3d_c4(out)
        # out = out + x4
        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c4(out)
        out += residual_4
        out = self.inorm3d_c4(out)
        out = self.lrelu(out)
        context_4 = out

        # Level 5

        out = self.conv3d_c5(out)
        # out = out + x5
        residual_5 = out
        out = self.norm_lrelu_conv_c5(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c5(out)
        out += residual_5
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)
        up_binear0 = nn.Upsample(size=(out.size(2), 2 * out.size(3), 2 * out.size(4)), mode='trilinear', align_corners=False)
        out = up_binear0(out)
        out = self.upl0(out)

        out = self.conv3d_l0(out)
        out = self.inorm3d_l0(out)
        out = self.lrelu(out)

        # Level 1 localization pathway
        out = torch.cat([out, context_4], dim=1)
        out = self.conv_norm_lrelu_l1(out)
        out = self.conv3d_l1(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)
        up_binear1 = nn.Upsample(size=(out.size(2), 2 * out.size(3), 2 * out.size(4)), mode='trilinear',align_corners=False)
        out = up_binear1(out)
        out = self.upl1(out)

        # Level 2 localization pathway
        out = torch.cat([out, context_3], dim=1)
        out = self.conv_norm_lrelu_l2(out)
        ds2 = out
        out = self.conv3d_l2(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)
        up_binear2 = nn.Upsample(size=(out.size(2), 2 * out.size(3), 2 * out.size(4)), mode='trilinear',align_corners=False)
        out = up_binear2(out)
        out = self.upl2(out)

        # Level 3 localization pathway
        out = torch.cat([out, context_2], dim=1)
        out = self.conv_norm_lrelu_l3(out)
        ds3 = out
        out = self.conv3d_l3(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)
        up_binear3 = nn.Upsample(size=(out.size(2), 2 * out.size(3), 2 * out.size(4)), mode='trilinear',align_corners=False)
        out = up_binear3(out)
        out = self.upl3(out)

        # Level 4 localization pathway
        out = torch.cat([out, context_1], dim=1)
        out = self.conv_norm_lrelu_l4(out)
        out_pred = self.conv3d_l4(out)

        ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
        up_binear4 = nn.Upsample(size=(ds2_1x1_conv.size(2), 2 * ds2_1x1_conv.size(3), 2 * ds2_1x1_conv.size(4)),
                                 mode='trilinear',align_corners=False)
        ds1_ds2_sum_upscale = up_binear4(ds2_1x1_conv)
        ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
        ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
        ds1_ds2_sum_upscale_ds3_sum_upscale = F.interpolate(ds1_ds2_sum_upscale_ds3_sum,
                                                         size=(ds1_ds2_sum_upscale_ds3_sum.size(2),
                                                               2 * ds1_ds2_sum_upscale_ds3_sum.size(3),
                                                               2 * ds1_ds2_sum_upscale_ds3_sum.size(4)),
                                                         mode='trilinear',align_corners=False)
        out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale
        out = F.interpolate(out, size=(out.size(2), 2*out.size(3), 2*out.size(4)), mode='trilinear',align_corners=False)
        return out

class UNet3D(nn.Module):
    def __init__(self, in_channel=1, n_classes=2, bn=False):
        self.in_channel = in_channel
        self.n_classes = n_classes
        super(UNet3D, self).__init__()
        self.ec0 = self.encoder(self.in_channel, 8, bias=False, batchnorm=bn)
        self.ec1 = self.encoder(8, 16, bias=False, batchnorm=bn)
        self.ec2 = self.encoder(16, 16, bias=False, batchnorm=bn)
        self.ec3 = self.encoder(16, 32, bias=False, batchnorm=bn)
        self.ec4 = self.encoder(32, 32, bias=False, batchnorm=bn)
        self.ec5 = self.encoder(32, 64, bias=False, batchnorm=bn)
        self.ec6 = self.encoder(64, 64, bias=False, batchnorm=bn)
        self.ec7 = self.encoder(64, 128, bias=False, batchnorm=bn)

        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)

        self.dc9 = self.decoder(128, 128, kernel_size=2, stride=2, bias=False)
        self.dc8 = self.decoder(64 + 128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc7 = self.decoder(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc6 = self.decoder(64, 64, kernel_size=2, stride=2, bias=False)
        self.dc5 = self.decoder(32 + 64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc4 = self.decoder(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc3 = self.decoder(32, 32, kernel_size=2, stride=2, bias=False)
        self.dc2 = self.decoder(16 + 32, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc1 = self.decoder(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc0 = self.decoder(16, n_classes, kernel_size=1, stride=1, bias=False)
        # self.dc9 = self.decoder(128, 128, kernel_size=3, stride=1, bias=False, padding=1)  # kernel_size=2, stride=2
        # self.dc8 = self.decoder(64 + 128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.dc7 = self.decoder(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.dc6 = self.decoder(64, 64, kernel_size=3, stride=1, bias=False, padding=1)  # kernel_size=2, stride=2
        # self.dc5 = self.decoder(32 + 64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        # self.dc4 = self.decoder(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        # self.dc3 = self.decoder(32, 32, kernel_size=3, stride=1, bias=False, padding=1)  # kernel_size=2, stride=2
        # self.dc2 = self.decoder(16 + 32, 16, kernel_size=3, stride=1, padding=1, bias=False)
        # self.dc1 = self.decoder(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        # self.dc0 = self.decoder(16, n_classes, kernel_size=1, stride=1, bias=False)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU())
        return layer


    def decoder(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            # nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU())
        return layer

    def forward(self, x):
        e0 = self.ec0(x)
        syn0 = self.ec1(e0)
        e1 = self.pool0(syn0)
        e2 = self.ec2(e1)
        syn1 = self.ec3(e2)
        del e0, e1, e2

        e3 = self.pool1(syn1)
        e4 = self.ec4(e3)
        syn2 = self.ec5(e4)
        del e3, e4

        e5 = self.pool2(syn2)
        e6 = self.ec6(e5)
        e7 = self.ec7(e6)     
        del e5, e6

        d9 = torch.cat((self.dc9(e7), syn2),dim=1)
        #d9 = torch.cat((self.dc9(F.interpolate(e7, scale_factor=2, mode="trilinear")), syn2), dim=1)
        del e7, syn2

        d8 = self.dc8(d9)
        d7 = self.dc7(d8)
        del d9, d8

        d6 = torch.cat((self.dc6(d7), syn1),dim=1)
        #d6 = torch.cat((self.dc6(F.interpolate(d7, scale_factor=2, mode="trilinear")), syn1), dim=1)
        del d7, syn1

        d5 = self.dc5(d6)
        d4 = self.dc4(d5)
        del d6, d5

        d3 = torch.cat((self.dc3(d4), syn0),dim=1)
        #d3 = torch.cat((self.dc3(F.interpolate(d4, scale_factor=2, mode="trilinear")), syn0), dim=1)
        del d4, syn0

        d2 = self.dc2(d3)
        d1 = self.dc1(d2)
        del d3, d2

        d0 = self.dc0(d1)
        return d0


