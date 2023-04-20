from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class UNet3D_MALA(nn.Module):
	def __init__(self, output_nc=3, if_sigmoid=True, init_mode='kaiming', show_feature=False):
		super(UNet3D_MALA, self).__init__()
		self.if_sigmoid = if_sigmoid
		self.init_mode = init_mode
		self.show_feature = show_feature
		self.conv1 = nn.Conv3d(1, 12, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv2 = nn.Conv3d(12, 12, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))
		
		self.conv3 = nn.Conv3d(12, 60, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv4 = nn.Conv3d(60, 60, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))
		
		self.conv5 = nn.Conv3d(60, 300, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv6 = nn.Conv3d(300, 300, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool3 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))
		
		self.conv7 = nn.Conv3d(300, 1500, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv8 = nn.Conv3d(1500, 1500, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		
		self.dconv1 = nn.ConvTranspose3d(1500, 1500, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=1500, bias=False)
		self.conv9 = nn.Conv3d(1500, 300, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv10 = nn.Conv3d(600, 300, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv11 = nn.Conv3d(300, 300, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		
		self.dconv2 = nn.ConvTranspose3d(300, 300, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=300, bias=False)
		self.conv12 = nn.Conv3d(300, 60, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv13 = nn.Conv3d(120, 60, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv14 = nn.Conv3d(60, 60, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		
		self.dconv3 = nn.ConvTranspose3d(60, 60, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=60, bias=False)
		self.conv15 = nn.Conv3d(60, 12, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv16 = nn.Conv3d(24, 12, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv17 = nn.Conv3d(12, 12, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		
		self.conv18 = nn.Conv3d(12, output_nc, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		# self.apply(self._weight_init)
		# Initialization
		for m in self.modules():
			if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
				if self.init_mode == 'kaiming':
					init.kaiming_normal_(m.weight, 0.005, 'fan_in', 'leaky_relu')
				elif self.init_mode == 'xavier':
					init.xavier_normal_(m.weight)
				elif self.init_mode == 'orthogonal':
					init.orthogonal_(m.weight)
				else:
					raise AttributeError('No this init mode!')
	
	@staticmethod
	def _weight_init(m):
		if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
			init.kaiming_normal_(m.weight, 0.005, 'fan_in', 'leaky_relu')
	
	def crop_and_concat(self, upsampled, bypass, crop=False):
		if crop:
			c = (bypass.size()[3] - upsampled.size()[3]) // 2
			cc = (bypass.size()[2] - upsampled.size()[2]) // 2
			assert(c > 0)
			assert(cc > 0)
			bypass = F.pad(bypass, (-c, -c, -c, -c, -cc, -cc))
		return torch.cat((upsampled, bypass), 1)
	
	def forward(self, input):
		conv1 = F.leaky_relu(self.conv1(input), 0.005)
		conv2 = F.leaky_relu(self.conv2(conv1), 0.005)
		pool1 = self.pool1(conv2)
		conv3 = F.leaky_relu(self.conv3(pool1), 0.005)
		conv4 = F.leaky_relu(self.conv4(conv3), 0.005)
		pool2 = self.pool2(conv4)
		conv5 = F.leaky_relu(self.conv5(pool2), 0.005)
		conv6 = F.leaky_relu(self.conv6(conv5), 0.005)
		pool3 = self.pool3(conv6)
		conv7 = F.leaky_relu(self.conv7(pool3), 0.005)
		conv8 = F.leaky_relu(self.conv8(conv7), 0.005)
		dconv1 = self.dconv1(conv8)
		conv9 = self.conv9(dconv1)
		mc1 = self.crop_and_concat(conv9, conv6, crop=True)
		conv10 = F.leaky_relu(self.conv10(mc1), 0.005)
		conv11 = F.leaky_relu(self.conv11(conv10), 0.005)
		dconv2 = self.dconv2(conv11)
		conv12 = self.conv12(dconv2)
		mc2 = self.crop_and_concat(conv12, conv4, crop=True)
		conv13 = F.leaky_relu(self.conv13(mc2), 0.005)
		conv14 = F.leaky_relu(self.conv14(conv13), 0.005)
		dconv3 = self.dconv3(conv14)
		conv15 = self.conv15(dconv3)
		mc3 = self.crop_and_concat(conv15, conv2, crop=True)
		conv16 = F.leaky_relu(self.conv16(mc3), 0.005)
		conv17 = F.leaky_relu(self.conv17(conv16), 0.005)
		output = self.conv18(conv17)
		if self.if_sigmoid:
			output = torch.sigmoid(output)
		if self.show_feature:
			return conv8, conv11, conv14, conv17, output
		else:
			return output


class UNet3D_MALA_small(nn.Module):
	def __init__(self, output_nc=3, if_sigmoid=True, init_mode='kaiming', show_feature=False):
		super(UNet3D_MALA_small, self).__init__()
		self.if_sigmoid = if_sigmoid
		self.init_mode = init_mode
		self.show_feature = show_feature
		self.conv1 = nn.Conv3d(1, 12, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv2 = nn.Conv3d(12, 12, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

		self.conv3 = nn.Conv3d(12, 60, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv4 = nn.Conv3d(60, 60, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

		self.conv5 = nn.Conv3d(60, 120, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv6 = nn.Conv3d(120, 120, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool3 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

		self.conv7 = nn.Conv3d(120, 300, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv8 = nn.Conv3d(300, 300, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.dconv1 = nn.ConvTranspose3d(300, 300, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=300,
										 bias=False)
		self.conv9 = nn.Conv3d(300, 120, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv10 = nn.Conv3d(240, 120, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv11 = nn.Conv3d(120, 120, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.dconv2 = nn.ConvTranspose3d(120, 120, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=120,
										 bias=False)
		self.conv12 = nn.Conv3d(120, 60, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv13 = nn.Conv3d(120, 60, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv14 = nn.Conv3d(60, 60, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.dconv3 = nn.ConvTranspose3d(60, 60, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=60,
										 bias=False)
		self.conv15 = nn.Conv3d(60, 12, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv16 = nn.Conv3d(24, 12, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv17 = nn.Conv3d(12, 12, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.conv18 = nn.Conv3d(12, output_nc, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		# self.apply(self._weight_init)
		# Initialization
		for m in self.modules():
			if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
				if self.init_mode == 'kaiming':
					init.kaiming_normal_(m.weight, 0.005, 'fan_in', 'leaky_relu')
				elif self.init_mode == 'xavier':
					init.xavier_normal_(m.weight)
				elif self.init_mode == 'orthogonal':
					init.orthogonal_(m.weight)
				else:
					raise AttributeError('No this init mode!')

	@staticmethod
	def _weight_init(m):
		if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
			init.kaiming_normal_(m.weight, 0.005, 'fan_in', 'leaky_relu')

	def crop_and_concat(self, upsampled, bypass, crop=False):
		if crop:
			c = (bypass.size()[3] - upsampled.size()[3]) // 2
			cc = (bypass.size()[2] - upsampled.size()[2]) // 2
			assert (c > 0)
			assert (cc > 0)
			bypass = F.pad(bypass, (-c, -c, -c, -c, -cc, -cc))
		return torch.cat((upsampled, bypass), 1)

	def forward(self, input):
		conv1 = F.leaky_relu(self.conv1(input), 0.005)
		conv2 = F.leaky_relu(self.conv2(conv1), 0.005)
		pool1 = self.pool1(conv2)
		conv3 = F.leaky_relu(self.conv3(pool1), 0.005)
		conv4 = F.leaky_relu(self.conv4(conv3), 0.005)
		pool2 = self.pool2(conv4)
		conv5 = F.leaky_relu(self.conv5(pool2), 0.005)
		conv6 = F.leaky_relu(self.conv6(conv5), 0.005)
		pool3 = self.pool3(conv6)
		conv7 = F.leaky_relu(self.conv7(pool3), 0.005)
		conv8 = F.leaky_relu(self.conv8(conv7), 0.005)
		dconv1 = self.dconv1(conv8)
		conv9 = self.conv9(dconv1)
		mc1 = self.crop_and_concat(conv9, conv6, crop=True)
		conv10 = F.leaky_relu(self.conv10(mc1), 0.005)
		conv11 = F.leaky_relu(self.conv11(conv10), 0.005)
		dconv2 = self.dconv2(conv11)
		conv12 = self.conv12(dconv2)
		mc2 = self.crop_and_concat(conv12, conv4, crop=True)
		conv13 = F.leaky_relu(self.conv13(mc2), 0.005)
		conv14 = F.leaky_relu(self.conv14(conv13), 0.005)
		dconv3 = self.dconv3(conv14)
		conv15 = self.conv15(dconv3)
		mc3 = self.crop_and_concat(conv15, conv2, crop=True)
		conv16 = F.leaky_relu(self.conv16(mc3), 0.005)
		conv17 = F.leaky_relu(self.conv17(conv16), 0.005)
		output = self.conv18(conv17)
		if self.if_sigmoid:
			output = torch.sigmoid(output)
		if self.show_feature:
			return conv8, conv11, conv14, conv17, output
		else:
			return output


class UNet3D_MALA_small2(nn.Module):
	def __init__(self, output_nc=3, if_sigmoid=True, init_mode='kaiming', show_feature=False):
		super(UNet3D_MALA_small2, self).__init__()
		self.if_sigmoid = if_sigmoid
		self.init_mode = init_mode
		self.show_feature = show_feature
		self.conv1 = nn.Conv3d(1, 6, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv2 = nn.Conv3d(6, 6, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

		self.conv3 = nn.Conv3d(6, 30, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv4 = nn.Conv3d(30, 30, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

		self.conv5 = nn.Conv3d(30, 150, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv6 = nn.Conv3d(150, 150, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool3 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

		self.conv7 = nn.Conv3d(150, 500, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv8 = nn.Conv3d(500, 500, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.dconv1 = nn.ConvTranspose3d(500, 500, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=500,
										 bias=False)
		self.conv9 = nn.Conv3d(500, 150, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv10 = nn.Conv3d(300, 150, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv11 = nn.Conv3d(150, 150, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.dconv2 = nn.ConvTranspose3d(150, 150, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=150,
										 bias=False)
		self.conv12 = nn.Conv3d(150, 30, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv13 = nn.Conv3d(60, 30, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv14 = nn.Conv3d(30, 30, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.dconv3 = nn.ConvTranspose3d(30, 30, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=30,
										 bias=False)
		self.conv15 = nn.Conv3d(30, 6, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv16 = nn.Conv3d(12, 6, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv17 = nn.Conv3d(6, 6, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.conv18 = nn.Conv3d(6, output_nc, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		# self.apply(self._weight_init)
		# Initialization
		for m in self.modules():
			if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
				if self.init_mode == 'kaiming':
					init.kaiming_normal_(m.weight, 0.005, 'fan_in', 'leaky_relu')
				elif self.init_mode == 'xavier':
					init.xavier_normal_(m.weight)
				elif self.init_mode == 'orthogonal':
					init.orthogonal_(m.weight)
				else:
					raise AttributeError('No this init mode!')

	@staticmethod
	def _weight_init(m):
		if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
			init.kaiming_normal_(m.weight, 0.005, 'fan_in', 'leaky_relu')

	def crop_and_concat(self, upsampled, bypass, crop=False):
		if crop:
			c = (bypass.size()[3] - upsampled.size()[3]) // 2
			cc = (bypass.size()[2] - upsampled.size()[2]) // 2
			assert (c > 0)
			assert (cc > 0)
			bypass = F.pad(bypass, (-c, -c, -c, -c, -cc, -cc))
		return torch.cat((upsampled, bypass), 1)

	def forward(self, input):
		conv1 = F.leaky_relu(self.conv1(input), 0.005)
		conv2 = F.leaky_relu(self.conv2(conv1), 0.005)
		pool1 = self.pool1(conv2)
		conv3 = F.leaky_relu(self.conv3(pool1), 0.005)
		conv4 = F.leaky_relu(self.conv4(conv3), 0.005)
		pool2 = self.pool2(conv4)
		conv5 = F.leaky_relu(self.conv5(pool2), 0.005)
		conv6 = F.leaky_relu(self.conv6(conv5), 0.005)
		pool3 = self.pool3(conv6)
		conv7 = F.leaky_relu(self.conv7(pool3), 0.005)
		conv8 = F.leaky_relu(self.conv8(conv7), 0.005)
		dconv1 = self.dconv1(conv8)
		conv9 = self.conv9(dconv1)
		mc1 = self.crop_and_concat(conv9, conv6, crop=True)
		conv10 = F.leaky_relu(self.conv10(mc1), 0.005)
		conv11 = F.leaky_relu(self.conv11(conv10), 0.005)
		dconv2 = self.dconv2(conv11)
		conv12 = self.conv12(dconv2)
		mc2 = self.crop_and_concat(conv12, conv4, crop=True)
		conv13 = F.leaky_relu(self.conv13(mc2), 0.005)
		conv14 = F.leaky_relu(self.conv14(conv13), 0.005)
		dconv3 = self.dconv3(conv14)
		conv15 = self.conv15(dconv3)
		mc3 = self.crop_and_concat(conv15, conv2, crop=True)
		conv16 = F.leaky_relu(self.conv16(mc3), 0.005)
		conv17 = F.leaky_relu(self.conv17(conv16), 0.005)
		output = self.conv18(conv17)
		if self.if_sigmoid:
			output = torch.sigmoid(output)
		if self.show_feature:
			return conv8, conv11, conv14, conv17, output
		else:
			return output



class UNet3D_MALA_small3(nn.Module):
	def __init__(self, output_nc=3, if_sigmoid=True, init_mode='kaiming', show_feature=False):
		super(UNet3D_MALA_small3, self).__init__()
		self.if_sigmoid = if_sigmoid
		self.init_mode = init_mode
		self.show_feature = show_feature
		self.conv1 = nn.Conv3d(1, 8, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv2 = nn.Conv3d(8, 8, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

		self.conv3 = nn.Conv3d(8, 16, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv4 = nn.Conv3d(16, 16, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

		self.conv5 = nn.Conv3d(16, 32, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv6 = nn.Conv3d(32, 32, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool3 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

		self.conv7 = nn.Conv3d(32, 64, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv8 = nn.Conv3d(64, 64, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.dconv1 = nn.ConvTranspose3d(64, 64, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=64,
										 bias=False)
		self.conv9 = nn.Conv3d(64, 32, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv10 = nn.Conv3d(64, 32, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv11 = nn.Conv3d(32, 32, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.dconv2 = nn.ConvTranspose3d(32, 32, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=32,
										 bias=False)
		self.conv12 = nn.Conv3d(32, 16, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv13 = nn.Conv3d(32, 32, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv14 = nn.Conv3d(32, 32, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.dconv3 = nn.ConvTranspose3d(32, 32, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=32,
										 bias=False)
		self.conv15 = nn.Conv3d(32, 8, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv16 = nn.Conv3d(16, 8, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv17 = nn.Conv3d(8, 8, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.conv18 = nn.Conv3d(8, output_nc, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		# self.apply(self._weight_init)
		# Initialization
		for m in self.modules():
			if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
				if self.init_mode == 'kaiming':
					init.kaiming_normal_(m.weight, 0.005, 'fan_in', 'leaky_relu')
				elif self.init_mode == 'xavier':
					init.xavier_normal_(m.weight)
				elif self.init_mode == 'orthogonal':
					init.orthogonal_(m.weight)
				else:
					raise AttributeError('No this init mode!')

	@staticmethod
	def _weight_init(m):
		if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
			init.kaiming_normal_(m.weight, 0.005, 'fan_in', 'leaky_relu')

	def crop_and_concat(self, upsampled, bypass, crop=False):
		if crop:
			c = (bypass.size()[3] - upsampled.size()[3]) // 2
			cc = (bypass.size()[2] - upsampled.size()[2]) // 2
			assert (c > 0)
			assert (cc > 0)
			bypass = F.pad(bypass, (-c, -c, -c, -c, -cc, -cc))
		return torch.cat((upsampled, bypass), 1)

	def forward(self, input):
		conv1 = F.leaky_relu(self.conv1(input), 0.005)
		conv2 = F.leaky_relu(self.conv2(conv1), 0.005)
		pool1 = self.pool1(conv2)
		conv3 = F.leaky_relu(self.conv3(pool1), 0.005)
		conv4 = F.leaky_relu(self.conv4(conv3), 0.005)
		pool2 = self.pool2(conv4)
		conv5 = F.leaky_relu(self.conv5(pool2), 0.005)
		conv6 = F.leaky_relu(self.conv6(conv5), 0.005)
		pool3 = self.pool3(conv6)
		conv7 = F.leaky_relu(self.conv7(pool3), 0.005)
		conv8 = F.leaky_relu(self.conv8(conv7), 0.005)
		dconv1 = self.dconv1(conv8)
		conv9 = self.conv9(dconv1)
		mc1 = self.crop_and_concat(conv9, conv6, crop=True)
		conv10 = F.leaky_relu(self.conv10(mc1), 0.005)
		conv11 = F.leaky_relu(self.conv11(conv10), 0.005)
		dconv2 = self.dconv2(conv11)
		conv12 = self.conv12(dconv2)
		mc2 = self.crop_and_concat(conv12, conv4, crop=True)
		conv13 = F.leaky_relu(self.conv13(mc2), 0.005)
		conv14 = F.leaky_relu(self.conv14(conv13), 0.005)
		dconv3 = self.dconv3(conv14)
		conv15 = self.conv15(dconv3)
		mc3 = self.crop_and_concat(conv15, conv2, crop=True)
		conv16 = F.leaky_relu(self.conv16(mc3), 0.005)
		conv17 = F.leaky_relu(self.conv17(conv16), 0.005)
		output = self.conv18(conv17)
		if self.if_sigmoid:
			output = torch.sigmoid(output)
		if self.show_feature:
			return conv8, conv11, conv14, conv17, output
		else:
			return output





class UNet3D_MALA_embedding(nn.Module):
	def __init__(self, output_nc=3, if_sigmoid=False, init_mode='kaiming', show_feature=False, emd=16):
		super(UNet3D_MALA_embedding, self).__init__()
		self.if_sigmoid = if_sigmoid
		self.init_mode = init_mode
		self.show_feature = show_feature
		self.emd = emd
		self.conv1 = nn.Conv3d(1, 12, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv2 = nn.Conv3d(12, 12, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))
		
		self.conv3 = nn.Conv3d(12, 60, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv4 = nn.Conv3d(60, 60, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))
		
		self.conv5 = nn.Conv3d(60, 300, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv6 = nn.Conv3d(300, 300, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool3 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))
		
		self.conv7 = nn.Conv3d(300, 1500, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv8 = nn.Conv3d(1500, 1500, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		
		self.dconv1 = nn.ConvTranspose3d(1500, 1500, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=1500, bias=False)
		self.conv9 = nn.Conv3d(1500, 300, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv10 = nn.Conv3d(600, 300, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv11 = nn.Conv3d(300, 300, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		
		self.dconv2 = nn.ConvTranspose3d(300, 300, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=300, bias=False)
		self.conv12 = nn.Conv3d(300, 60, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv13 = nn.Conv3d(120, 60, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv14 = nn.Conv3d(60, 60, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		
		self.dconv3 = nn.ConvTranspose3d(60, 60, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=60, bias=False)
		self.conv15 = nn.Conv3d(60, 12, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv16 = nn.Conv3d(24, 12, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv17 = nn.Conv3d(12, 12, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		
		self.conv18 = nn.Conv3d(12, self.emd, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		# self.apply(self._weight_init)
		# Initialization
		for m in self.modules():
			if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
				if self.init_mode == 'kaiming':
					init.kaiming_normal_(m.weight, 0.005, 'fan_in', 'leaky_relu')
				elif self.init_mode == 'xavier':
					init.xavier_normal_(m.weight)
				elif self.init_mode == 'orthogonal':
					init.orthogonal_(m.weight)
				else:
					raise AttributeError('No this init mode!')
	
	@staticmethod
	def _weight_init(m):
		if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
			init.kaiming_normal_(m.weight, 0.005, 'fan_in', 'leaky_relu')
	
	def crop_and_concat(self, upsampled, bypass, crop=False):
		if crop:
			c = (bypass.size()[3] - upsampled.size()[3]) // 2
			cc = (bypass.size()[2] - upsampled.size()[2]) // 2
			assert(c > 0)
			assert(cc > 0)
			bypass = F.pad(bypass, (-c, -c, -c, -c, -cc, -cc))
		return torch.cat((upsampled, bypass), 1)
	
	def forward(self, input):
		conv1 = F.leaky_relu(self.conv1(input), 0.005)
		conv2 = F.leaky_relu(self.conv2(conv1), 0.005)
		pool1 = self.pool1(conv2)
		conv3 = F.leaky_relu(self.conv3(pool1), 0.005)
		conv4 = F.leaky_relu(self.conv4(conv3), 0.005)
		pool2 = self.pool2(conv4)
		conv5 = F.leaky_relu(self.conv5(pool2), 0.005)
		conv6 = F.leaky_relu(self.conv6(conv5), 0.005)
		pool3 = self.pool3(conv6)
		conv7 = F.leaky_relu(self.conv7(pool3), 0.005)
		conv8 = F.leaky_relu(self.conv8(conv7), 0.005)
		dconv1 = self.dconv1(conv8)
		conv9 = self.conv9(dconv1)
		mc1 = self.crop_and_concat(conv9, conv6, crop=True)
		conv10 = F.leaky_relu(self.conv10(mc1), 0.005)
		conv11 = F.leaky_relu(self.conv11(conv10), 0.005)
		dconv2 = self.dconv2(conv11)
		conv12 = self.conv12(dconv2)
		mc2 = self.crop_and_concat(conv12, conv4, crop=True)
		conv13 = F.leaky_relu(self.conv13(mc2), 0.005)
		conv14 = F.leaky_relu(self.conv14(conv13), 0.005)
		dconv3 = self.dconv3(conv14)
		conv15 = self.conv15(dconv3)
		mc3 = self.crop_and_concat(conv15, conv2, crop=True)
		conv16 = F.leaky_relu(self.conv16(mc3), 0.005)
		conv17 = F.leaky_relu(self.conv17(conv16), 0.005)
		output = self.conv18(conv17)
		# if self.if_sigmoid:
		# 	output = torch.sigmoid(output)
		if self.show_feature:
			return conv8, conv11, conv14, conv17, output
		# else:
		# 	return output
		return output


class UNet3D_MALA_embedding_small(nn.Module):
	def __init__(self, output_nc=3, if_sigmoid=False, init_mode='kaiming', show_feature=False, emd=16):
		super(UNet3D_MALA_embedding_small, self).__init__()
		self.if_sigmoid = if_sigmoid
		self.init_mode = init_mode
		self.show_feature = show_feature
		self.emd = emd
		self.conv1 = nn.Conv3d(1, 12, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv2 = nn.Conv3d(12, 12, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

		self.conv3 = nn.Conv3d(12, 60, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv4 = nn.Conv3d(60, 60, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

		self.conv5 = nn.Conv3d(60, 120, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv6 = nn.Conv3d(120, 120, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool3 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

		self.conv7 = nn.Conv3d(120, 300, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv8 = nn.Conv3d(300, 300, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.dconv1 = nn.ConvTranspose3d(300, 300, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=300,
										 bias=False)
		self.conv9 = nn.Conv3d(300, 120, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv10 = nn.Conv3d(240, 120, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv11 = nn.Conv3d(120, 120, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.dconv2 = nn.ConvTranspose3d(120, 120, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=120,
										 bias=False)
		self.conv12 = nn.Conv3d(120, 60, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv13 = nn.Conv3d(120, 60, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv14 = nn.Conv3d(60, 60, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.dconv3 = nn.ConvTranspose3d(60, 60, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=60,
										 bias=False)
		self.conv15 = nn.Conv3d(60, 12, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv16 = nn.Conv3d(24, 12, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv17 = nn.Conv3d(12, 12, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.conv18 = nn.Conv3d(12, self.emd, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		# self.apply(self._weight_init)
		# Initialization
		for m in self.modules():
			if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
				if self.init_mode == 'kaiming':
					init.kaiming_normal_(m.weight, 0.005, 'fan_in', 'leaky_relu')
				elif self.init_mode == 'xavier':
					init.xavier_normal_(m.weight)
				elif self.init_mode == 'orthogonal':
					init.orthogonal_(m.weight)
				else:
					raise AttributeError('No this init mode!')

	@staticmethod
	def _weight_init(m):
		if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
			init.kaiming_normal_(m.weight, 0.005, 'fan_in', 'leaky_relu')

	def crop_and_concat(self, upsampled, bypass, crop=False):
		if crop:
			c = (bypass.size()[3] - upsampled.size()[3]) // 2
			cc = (bypass.size()[2] - upsampled.size()[2]) // 2
			assert (c > 0)
			assert (cc > 0)
			bypass = F.pad(bypass, (-c, -c, -c, -c, -cc, -cc))
		return torch.cat((upsampled, bypass), 1)

	def forward(self, input):
		conv1 = F.leaky_relu(self.conv1(input), 0.005)
		conv2 = F.leaky_relu(self.conv2(conv1), 0.005)
		pool1 = self.pool1(conv2)
		conv3 = F.leaky_relu(self.conv3(pool1), 0.005)
		conv4 = F.leaky_relu(self.conv4(conv3), 0.005)
		pool2 = self.pool2(conv4)
		conv5 = F.leaky_relu(self.conv5(pool2), 0.005)
		conv6 = F.leaky_relu(self.conv6(conv5), 0.005)
		pool3 = self.pool3(conv6)
		conv7 = F.leaky_relu(self.conv7(pool3), 0.005)
		conv8 = F.leaky_relu(self.conv8(conv7), 0.005)
		dconv1 = self.dconv1(conv8)
		conv9 = self.conv9(dconv1)
		mc1 = self.crop_and_concat(conv9, conv6, crop=True)
		conv10 = F.leaky_relu(self.conv10(mc1), 0.005)
		conv11 = F.leaky_relu(self.conv11(conv10), 0.005)
		dconv2 = self.dconv2(conv11)
		conv12 = self.conv12(dconv2)
		mc2 = self.crop_and_concat(conv12, conv4, crop=True)
		conv13 = F.leaky_relu(self.conv13(mc2), 0.005)
		conv14 = F.leaky_relu(self.conv14(conv13), 0.005)
		dconv3 = self.dconv3(conv14)
		conv15 = self.conv15(dconv3)
		mc3 = self.crop_and_concat(conv15, conv2, crop=True)
		conv16 = F.leaky_relu(self.conv16(mc3), 0.005)
		conv17 = F.leaky_relu(self.conv17(conv16), 0.005)
		output = self.conv18(conv17)
		# if self.if_sigmoid:
		# 	output = torch.sigmoid(output)
		# if self.show_feature:
		# 	return conv8, conv11, conv14, conv17, output
		# else:
		# 	return output
		return output



class UNet3D_MALA_embedding_small1(nn.Module):
	def __init__(self, output_nc=3, if_sigmoid=False, init_mode='kaiming', show_feature=False, emd=16):
		super(UNet3D_MALA_embedding_small1, self).__init__()
		self.if_sigmoid = if_sigmoid
		self.init_mode = init_mode
		self.show_feature = show_feature
		self.emd = emd
		self.conv1 = nn.Conv3d(1, 8, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv2 = nn.Conv3d(8, 8, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

		self.conv3 = nn.Conv3d(8, 16, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv4 = nn.Conv3d(16, 16, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

		self.conv5 = nn.Conv3d(16, 32, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv6 = nn.Conv3d(32, 32, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool3 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

		self.conv7 = nn.Conv3d(32, 64, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv8 = nn.Conv3d(64, 64, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.dconv1 = nn.ConvTranspose3d(64, 64, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=64,
										 bias=False)
		self.conv9 = nn.Conv3d(64, 32, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv10 = nn.Conv3d(64, 32, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv11 = nn.Conv3d(32, 32, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.dconv2 = nn.ConvTranspose3d(32, 32, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=32,
										 bias=False)
		self.conv12 = nn.Conv3d(32, 16, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv13 = nn.Conv3d(32, 32, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv14 = nn.Conv3d(32, 32, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.dconv3 = nn.ConvTranspose3d(32, 32, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=32,
										 bias=False)
		self.conv15 = nn.Conv3d(32, 8, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv16 = nn.Conv3d(16, 8, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv17 = nn.Conv3d(8, 8, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.conv18 = nn.Conv3d(8, self.emd, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		# self.apply(self._weight_init)
		# Initialization
		for m in self.modules():
			if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
				if self.init_mode == 'kaiming':
					init.kaiming_normal_(m.weight, 0.005, 'fan_in', 'leaky_relu')
				elif self.init_mode == 'xavier':
					init.xavier_normal_(m.weight)
				elif self.init_mode == 'orthogonal':
					init.orthogonal_(m.weight)
				else:
					raise AttributeError('No this init mode!')

	@staticmethod
	def _weight_init(m):
		if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
			init.kaiming_normal_(m.weight, 0.005, 'fan_in', 'leaky_relu')

	def crop_and_concat(self, upsampled, bypass, crop=False):
		if crop:
			c = (bypass.size()[3] - upsampled.size()[3]) // 2
			cc = (bypass.size()[2] - upsampled.size()[2]) // 2
			assert (c > 0)
			assert (cc > 0)
			bypass = F.pad(bypass, (-c, -c, -c, -c, -cc, -cc))
		return torch.cat((upsampled, bypass), 1)

	def forward(self, input):
		conv1 = F.leaky_relu(self.conv1(input), 0.005)
		conv2 = F.leaky_relu(self.conv2(conv1), 0.005)
		pool1 = self.pool1(conv2)
		conv3 = F.leaky_relu(self.conv3(pool1), 0.005)
		conv4 = F.leaky_relu(self.conv4(conv3), 0.005)
		pool2 = self.pool2(conv4)
		conv5 = F.leaky_relu(self.conv5(pool2), 0.005)
		conv6 = F.leaky_relu(self.conv6(conv5), 0.005)
		pool3 = self.pool3(conv6)
		conv7 = F.leaky_relu(self.conv7(pool3), 0.005)
		conv8 = F.leaky_relu(self.conv8(conv7), 0.005)
		dconv1 = self.dconv1(conv8)
		conv9 = self.conv9(dconv1)
		mc1 = self.crop_and_concat(conv9, conv6, crop=True)
		conv10 = F.leaky_relu(self.conv10(mc1), 0.005)
		conv11 = F.leaky_relu(self.conv11(conv10), 0.005)
		dconv2 = self.dconv2(conv11)
		conv12 = self.conv12(dconv2)
		mc2 = self.crop_and_concat(conv12, conv4, crop=True)
		conv13 = F.leaky_relu(self.conv13(mc2), 0.005)
		conv14 = F.leaky_relu(self.conv14(conv13), 0.005)
		dconv3 = self.dconv3(conv14)
		conv15 = self.conv15(dconv3)
		mc3 = self.crop_and_concat(conv15, conv2, crop=True)
		conv16 = F.leaky_relu(self.conv16(mc3), 0.005)
		conv17 = F.leaky_relu(self.conv17(conv16), 0.005)
		output = self.conv18(conv17)
		# if self.if_sigmoid:
		# 	output = torch.sigmoid(output)
		if self.show_feature:
			return conv8, conv11, conv14, conv17, output
		# else:
		# 	return output
		return output


class UNet3D_MALA_embedding_small2(nn.Module):
	def __init__(self, output_nc=3, if_sigmoid=False, init_mode='kaiming', show_feature=False, emd=16):
		super(UNet3D_MALA_embedding_small2, self).__init__()
		self.if_sigmoid = if_sigmoid
		self.init_mode = init_mode
		self.show_feature = show_feature
		self.emd = emd
		self.conv1 = nn.Conv3d(1, 6, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv2 = nn.Conv3d(6, 6, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

		self.conv3 = nn.Conv3d(6, 30, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv4 = nn.Conv3d(30, 30, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

		self.conv5 = nn.Conv3d(30, 150, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv6 = nn.Conv3d(150, 150, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool3 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

		self.conv7 = nn.Conv3d(150, 500, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv8 = nn.Conv3d(500, 500, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.dconv1 = nn.ConvTranspose3d(500, 500, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=500,
										 bias=False)
		self.conv9 = nn.Conv3d(500, 150, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv10 = nn.Conv3d(300, 150, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv11 = nn.Conv3d(150, 150, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.dconv2 = nn.ConvTranspose3d(150, 150, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=150,
										 bias=False)
		self.conv12 = nn.Conv3d(150, 30, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv13 = nn.Conv3d(60, 30, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv14 = nn.Conv3d(30, 30, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.dconv3 = nn.ConvTranspose3d(30, 30, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=30,
										 bias=False)
		self.conv15 = nn.Conv3d(30, 6, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv16 = nn.Conv3d(12, 6, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv17 = nn.Conv3d(6, 6, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.conv18 = nn.Conv3d(6, self.emd, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		# self.apply(self._weight_init)
		# Initialization
		for m in self.modules():
			if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
				if self.init_mode == 'kaiming':
					init.kaiming_normal_(m.weight, 0.005, 'fan_in', 'leaky_relu')
				elif self.init_mode == 'xavier':
					init.xavier_normal_(m.weight)
				elif self.init_mode == 'orthogonal':
					init.orthogonal_(m.weight)
				else:
					raise AttributeError('No this init mode!')

	@staticmethod
	def _weight_init(m):
		if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
			init.kaiming_normal_(m.weight, 0.005, 'fan_in', 'leaky_relu')

	def crop_and_concat(self, upsampled, bypass, crop=False):
		if crop:
			c = (bypass.size()[3] - upsampled.size()[3]) // 2
			cc = (bypass.size()[2] - upsampled.size()[2]) // 2
			assert (c > 0)
			assert (cc > 0)
			bypass = F.pad(bypass, (-c, -c, -c, -c, -cc, -cc))
		return torch.cat((upsampled, bypass), 1)

	def forward(self, input):
		conv1 = F.leaky_relu(self.conv1(input), 0.005)
		conv2 = F.leaky_relu(self.conv2(conv1), 0.005)
		pool1 = self.pool1(conv2)
		conv3 = F.leaky_relu(self.conv3(pool1), 0.005)
		conv4 = F.leaky_relu(self.conv4(conv3), 0.005)
		pool2 = self.pool2(conv4)
		conv5 = F.leaky_relu(self.conv5(pool2), 0.005)
		conv6 = F.leaky_relu(self.conv6(conv5), 0.005)
		pool3 = self.pool3(conv6)
		conv7 = F.leaky_relu(self.conv7(pool3), 0.005)
		conv8 = F.leaky_relu(self.conv8(conv7), 0.005)
		dconv1 = self.dconv1(conv8)
		conv9 = self.conv9(dconv1)
		mc1 = self.crop_and_concat(conv9, conv6, crop=True)
		conv10 = F.leaky_relu(self.conv10(mc1), 0.005)
		conv11 = F.leaky_relu(self.conv11(conv10), 0.005)
		dconv2 = self.dconv2(conv11)
		conv12 = self.conv12(dconv2)
		mc2 = self.crop_and_concat(conv12, conv4, crop=True)
		conv13 = F.leaky_relu(self.conv13(mc2), 0.005)
		conv14 = F.leaky_relu(self.conv14(conv13), 0.005)
		dconv3 = self.dconv3(conv14)
		conv15 = self.conv15(dconv3)
		mc3 = self.crop_and_concat(conv15, conv2, crop=True)
		conv16 = F.leaky_relu(self.conv16(mc3), 0.005)
		conv17 = F.leaky_relu(self.conv17(conv16), 0.005)
		output = self.conv18(conv17)
		# if self.if_sigmoid:
		# 	output = torch.sigmoid(output)
		if self.show_feature:
			return conv8, conv11, conv14, conv17, output
		# else:
		# 	return output
		return output


class UNet3D_MALA_embedding_small3(nn.Module):
	def __init__(self, output_nc=3, if_sigmoid=False, init_mode='kaiming', show_feature=False, emd=16):
		super(UNet3D_MALA_embedding_small3, self).__init__()
		self.if_sigmoid = if_sigmoid
		self.init_mode = init_mode
		self.show_feature = show_feature
		self.emd = emd
		self.conv1 = nn.Conv3d(1, 8, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv2 = nn.Conv3d(8, 8, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

		self.conv3 = nn.Conv3d(8, 16, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv4 = nn.Conv3d(16, 16, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

		self.conv5 = nn.Conv3d(16, 32, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv6 = nn.Conv3d(32, 32, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool3 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

		self.conv7 = nn.Conv3d(32, 64, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv8 = nn.Conv3d(64, 64, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.dconv1 = nn.ConvTranspose3d(64, 64, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=64,
										 bias=False)
		self.conv9 = nn.Conv3d(64, 32, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv10 = nn.Conv3d(64, 32, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv11 = nn.Conv3d(32, 32, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.dconv2 = nn.ConvTranspose3d(32, 32, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=32,
										 bias=False)
		self.conv12 = nn.Conv3d(32, 16, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv13 = nn.Conv3d(32, 32, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv14 = nn.Conv3d(32, 32, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.dconv3 = nn.ConvTranspose3d(32, 32, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=32,
										 bias=False)
		self.conv15 = nn.Conv3d(32, 8, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv16 = nn.Conv3d(16, 8, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv17 = nn.Conv3d(8, 8, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.conv18 = nn.Conv3d(8, self.emd, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		# self.apply(self._weight_init)
		# Initialization
		for m in self.modules():
			if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
				if self.init_mode == 'kaiming':
					init.kaiming_normal_(m.weight, 0.005, 'fan_in', 'leaky_relu')
				elif self.init_mode == 'xavier':
					init.xavier_normal_(m.weight)
				elif self.init_mode == 'orthogonal':
					init.orthogonal_(m.weight)
				else:
					raise AttributeError('No this init mode!')

	@staticmethod
	def _weight_init(m):
		if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
			init.kaiming_normal_(m.weight, 0.005, 'fan_in', 'leaky_relu')

	def crop_and_concat(self, upsampled, bypass, crop=False):
		if crop:
			c = (bypass.size()[3] - upsampled.size()[3]) // 2
			cc = (bypass.size()[2] - upsampled.size()[2]) // 2
			assert (c > 0)
			assert (cc > 0)
			bypass = F.pad(bypass, (-c, -c, -c, -c, -cc, -cc))
		return torch.cat((upsampled, bypass), 1)

	def forward(self, input):
		conv1 = F.leaky_relu(self.conv1(input), 0.005)
		conv2 = F.leaky_relu(self.conv2(conv1), 0.005)
		pool1 = self.pool1(conv2)
		conv3 = F.leaky_relu(self.conv3(pool1), 0.005)
		conv4 = F.leaky_relu(self.conv4(conv3), 0.005)
		pool2 = self.pool2(conv4)
		conv5 = F.leaky_relu(self.conv5(pool2), 0.005)
		conv6 = F.leaky_relu(self.conv6(conv5), 0.005)
		pool3 = self.pool3(conv6)
		conv7 = F.leaky_relu(self.conv7(pool3), 0.005)
		conv8 = F.leaky_relu(self.conv8(conv7), 0.005)
		dconv1 = self.dconv1(conv8)
		conv9 = self.conv9(dconv1)
		mc1 = self.crop_and_concat(conv9, conv6, crop=True)
		conv10 = F.leaky_relu(self.conv10(mc1), 0.005)
		conv11 = F.leaky_relu(self.conv11(conv10), 0.005)
		dconv2 = self.dconv2(conv11)
		conv12 = self.conv12(dconv2)
		mc2 = self.crop_and_concat(conv12, conv4, crop=True)
		conv13 = F.leaky_relu(self.conv13(mc2), 0.005)
		conv14 = F.leaky_relu(self.conv14(conv13), 0.005)
		dconv3 = self.dconv3(conv14)
		conv15 = self.conv15(dconv3)
		mc3 = self.crop_and_concat(conv15, conv2, crop=True)
		conv16 = F.leaky_relu(self.conv16(mc3), 0.005)
		conv17 = F.leaky_relu(self.conv17(conv16), 0.005)
		output = self.conv18(conv17)
		# if self.if_sigmoid:
		# 	output = torch.sigmoid(output)
		if self.show_feature:
			return conv8, conv11, conv14, conv17, output
		# else:
		# 	return output
		return output




class UNet3D_MALA_embedding_small4(nn.Module):
	def __init__(self, output_nc=3, if_sigmoid=False, init_mode='kaiming', show_feature=False, emd=16):
		super(UNet3D_MALA_embedding_small4, self).__init__()
		self.if_sigmoid = if_sigmoid
		self.init_mode = init_mode
		self.show_feature = show_feature
		self.emd = emd
		self.conv1 = nn.Conv3d(1, 10, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv2 = nn.Conv3d(10, 10, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)  #8-12
		self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

		self.conv3 = nn.Conv3d(10, 20, 3, stride=1, padding=0, dilation=1, groups=1, bias=True) #16-24
		self.conv4 = nn.Conv3d(20, 20, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

		self.conv5 = nn.Conv3d(20, 40, 3, stride=1, padding=0, dilation=1, groups=1, bias=True) #32-48
		self.conv6 = nn.Conv3d(40, 40, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool3 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

		self.conv7 = nn.Conv3d(40, 80, 3, stride=1, padding=0, dilation=1, groups=1, bias=True) #64-96
		self.conv8 = nn.Conv3d(80, 80, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.dconv1 = nn.ConvTranspose3d(80, 80, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=80,
										 bias=False)
		self.conv9 = nn.Conv3d(80, 40, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv10 = nn.Conv3d(80, 40, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv11 = nn.Conv3d(40, 40, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.dconv2 = nn.ConvTranspose3d(40, 40, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=40,
										 bias=False)
		self.conv12 = nn.Conv3d(40, 20, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv13 = nn.Conv3d(40, 20, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv14 = nn.Conv3d(20, 20, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.dconv3 = nn.ConvTranspose3d(20, 20, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=20,
										 bias=False)
		self.conv15 = nn.Conv3d(20, 10, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv16 = nn.Conv3d(20, 10, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv17 = nn.Conv3d(10, 10, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.conv18 = nn.Conv3d(10, self.emd, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		# self.apply(self._weight_init)
		# Initialization
		for m in self.modules():
			if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
				if self.init_mode == 'kaiming':
					init.kaiming_normal_(m.weight, 0.005, 'fan_in', 'leaky_relu')
				elif self.init_mode == 'xavier':
					init.xavier_normal_(m.weight)
				elif self.init_mode == 'orthogonal':
					init.orthogonal_(m.weight)
				else:
					raise AttributeError('No this init mode!')

	@staticmethod
	def _weight_init(m):
		if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
			init.kaiming_normal_(m.weight, 0.005, 'fan_in', 'leaky_relu')

	def crop_and_concat(self, upsampled, bypass, crop=False):
		if crop:
			c = (bypass.size()[3] - upsampled.size()[3]) // 2
			cc = (bypass.size()[2] - upsampled.size()[2]) // 2
			assert (c > 0)
			assert (cc > 0)
			bypass = F.pad(bypass, (-c, -c, -c, -c, -cc, -cc))
		return torch.cat((upsampled, bypass), 1)

	def forward(self, input):
		conv1 = F.leaky_relu(self.conv1(input), 0.005)
		conv2 = F.leaky_relu(self.conv2(conv1), 0.005)
		pool1 = self.pool1(conv2)
		conv3 = F.leaky_relu(self.conv3(pool1), 0.005)
		conv4 = F.leaky_relu(self.conv4(conv3), 0.005)
		pool2 = self.pool2(conv4)
		conv5 = F.leaky_relu(self.conv5(pool2), 0.005)
		conv6 = F.leaky_relu(self.conv6(conv5), 0.005)
		pool3 = self.pool3(conv6)
		conv7 = F.leaky_relu(self.conv7(pool3), 0.005)
		conv8 = F.leaky_relu(self.conv8(conv7), 0.005)
		dconv1 = self.dconv1(conv8)
		conv9 = self.conv9(dconv1)
		mc1 = self.crop_and_concat(conv9, conv6, crop=True)
		conv10 = F.leaky_relu(self.conv10(mc1), 0.005)
		conv11 = F.leaky_relu(self.conv11(conv10), 0.005)
		dconv2 = self.dconv2(conv11)
		conv12 = self.conv12(dconv2)
		mc2 = self.crop_and_concat(conv12, conv4, crop=True)
		conv13 = F.leaky_relu(self.conv13(mc2), 0.005)
		conv14 = F.leaky_relu(self.conv14(conv13), 0.005)
		dconv3 = self.dconv3(conv14)
		conv15 = self.conv15(dconv3)
		mc3 = self.crop_and_concat(conv15, conv2, crop=True)
		conv16 = F.leaky_relu(self.conv16(mc3), 0.005)
		conv17 = F.leaky_relu(self.conv17(conv16), 0.005)
		output = self.conv18(conv17)
		# if self.if_sigmoid:
		# 	output = torch.sigmoid(output)
		if self.show_feature:
			return conv8, conv11, conv14, conv17, output
		# else:
		# 	return output
		return output



class UNet3D_MALA_embedding_small5(nn.Module):
	def __init__(self, output_nc=3, if_sigmoid=False, init_mode='kaiming', show_feature=False, emd=16):
		super(UNet3D_MALA_embedding_small5, self).__init__()
		self.if_sigmoid = if_sigmoid
		self.init_mode = init_mode
		self.show_feature = show_feature
		self.emd = emd
		self.conv1 = nn.Conv3d(1, 12, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv2 = nn.Conv3d(12, 12, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)  #8-12
		self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

		self.conv3 = nn.Conv3d(12, 24, 3, stride=1, padding=0, dilation=1, groups=1, bias=True) #16-24
		self.conv4 = nn.Conv3d(24, 24, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

		self.conv5 = nn.Conv3d(24, 48, 3, stride=1, padding=0, dilation=1, groups=1, bias=True) #32-48
		self.conv6 = nn.Conv3d(48, 48, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool3 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

		self.conv7 = nn.Conv3d(48, 96, 3, stride=1, padding=0, dilation=1, groups=1, bias=True) #64-96
		self.conv8 = nn.Conv3d(96, 96, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.dconv1 = nn.ConvTranspose3d(96, 96, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=96,
										 bias=False)
		self.conv9 = nn.Conv3d(96, 48, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv10 = nn.Conv3d(96, 48, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv11 = nn.Conv3d(48, 48, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.dconv2 = nn.ConvTranspose3d(48, 48, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=48,
										 bias=False)
		self.conv12 = nn.Conv3d(48, 24, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv13 = nn.Conv3d(48, 24, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv14 = nn.Conv3d(24, 24, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.dconv3 = nn.ConvTranspose3d(24, 24, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=24,
										 bias=False)
		self.conv15 = nn.Conv3d(24, 12, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv16 = nn.Conv3d(24, 12, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv17 = nn.Conv3d(12, 12, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.conv18 = nn.Conv3d(12, self.emd, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		# self.apply(self._weight_init)
		# Initialization
		for m in self.modules():
			if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
				if self.init_mode == 'kaiming':
					init.kaiming_normal_(m.weight, 0.005, 'fan_in', 'leaky_relu')
				elif self.init_mode == 'xavier':
					init.xavier_normal_(m.weight)
				elif self.init_mode == 'orthogonal':
					init.orthogonal_(m.weight)
				else:
					raise AttributeError('No this init mode!')

	@staticmethod
	def _weight_init(m):
		if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
			init.kaiming_normal_(m.weight, 0.005, 'fan_in', 'leaky_relu')

	def crop_and_concat(self, upsampled, bypass, crop=False):
		if crop:
			c = (bypass.size()[3] - upsampled.size()[3]) // 2
			cc = (bypass.size()[2] - upsampled.size()[2]) // 2
			assert (c > 0)
			assert (cc > 0)
			bypass = F.pad(bypass, (-c, -c, -c, -c, -cc, -cc))
		return torch.cat((upsampled, bypass), 1)

	def forward(self, input):
		conv1 = F.leaky_relu(self.conv1(input), 0.005)
		conv2 = F.leaky_relu(self.conv2(conv1), 0.005)
		pool1 = self.pool1(conv2)
		conv3 = F.leaky_relu(self.conv3(pool1), 0.005)
		conv4 = F.leaky_relu(self.conv4(conv3), 0.005)
		pool2 = self.pool2(conv4)
		conv5 = F.leaky_relu(self.conv5(pool2), 0.005)
		conv6 = F.leaky_relu(self.conv6(conv5), 0.005)
		pool3 = self.pool3(conv6)
		conv7 = F.leaky_relu(self.conv7(pool3), 0.005)
		conv8 = F.leaky_relu(self.conv8(conv7), 0.005)
		dconv1 = self.dconv1(conv8)
		conv9 = self.conv9(dconv1)
		mc1 = self.crop_and_concat(conv9, conv6, crop=True)
		conv10 = F.leaky_relu(self.conv10(mc1), 0.005)
		conv11 = F.leaky_relu(self.conv11(conv10), 0.005)
		dconv2 = self.dconv2(conv11)
		conv12 = self.conv12(dconv2)
		mc2 = self.crop_and_concat(conv12, conv4, crop=True)
		conv13 = F.leaky_relu(self.conv13(mc2), 0.005)
		conv14 = F.leaky_relu(self.conv14(conv13), 0.005)
		dconv3 = self.dconv3(conv14)
		conv15 = self.conv15(dconv3)
		mc3 = self.crop_and_concat(conv15, conv2, crop=True)
		conv16 = F.leaky_relu(self.conv16(mc3), 0.005)
		conv17 = F.leaky_relu(self.conv17(conv16), 0.005)
		output = self.conv18(conv17)
		# if self.if_sigmoid:
		# 	output = torch.sigmoid(output)
		if self.show_feature:
			return conv8, conv11, conv14, conv17, output
		# else:
		# 	return output
		return output


class UNet3D_MALA_deep(nn.Module):
	def __init__(self, output_nc=3, if_sigmoid=False, init_mode='kaiming', show_feature=False, emd=16):
		super(UNet3D_MALA_deep, self).__init__()
		self.if_sigmoid = if_sigmoid
		self.init_mode = init_mode
		self.show_feature = show_feature
		self.emd = emd
		self.conv1 = nn.Conv3d(1, 12, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv2 = nn.Conv3d(12, 12, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))
		
		self.conv3 = nn.Conv3d(12, 60, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv4 = nn.Conv3d(60, 60, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))
		
		self.conv5 = nn.Conv3d(60, 300, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv6 = nn.Conv3d(300, 300, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool3 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))
		
		self.conv7 = nn.Conv3d(300, 1500, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv8 = nn.Conv3d(1500, 1500, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		
		self.dconv1 = nn.ConvTranspose3d(1500, 1500, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=1500, bias=False)
		self.conv9 = nn.Conv3d(1500, 300, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv10 = nn.Conv3d(600, 300, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv11 = nn.Conv3d(300, 300, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		
		self.dconv2 = nn.ConvTranspose3d(300, 300, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=300, bias=False)
		self.conv12 = nn.Conv3d(300, 60, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv13 = nn.Conv3d(120, 60, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv14 = nn.Conv3d(60, 60, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		
		self.dconv3 = nn.ConvTranspose3d(60, 60, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=60, bias=False)
		self.conv15 = nn.Conv3d(60, 12, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv16 = nn.Conv3d(24, 12, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv17 = nn.Conv3d(12, 12, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		
		self.conv18 = nn.Conv3d(12, self.emd, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		# self.apply(self._weight_init)
		# Initialization
		for m in self.modules():
			if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
				if self.init_mode == 'kaiming':
					init.kaiming_normal_(m.weight, 0.005, 'fan_in', 'leaky_relu')
				elif self.init_mode == 'xavier':
					init.xavier_normal_(m.weight)
				elif self.init_mode == 'orthogonal':
					init.orthogonal_(m.weight)
				else:
					raise AttributeError('No this init mode!')
	
	@staticmethod
	def _weight_init(m):
		if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
			init.kaiming_normal_(m.weight, 0.005, 'fan_in', 'leaky_relu')
	
	def crop_and_concat(self, upsampled, bypass, crop=False):
		if crop:
			c = (bypass.size()[3] - upsampled.size()[3]) // 2
			cc = (bypass.size()[2] - upsampled.size()[2]) // 2
			assert(c > 0)
			assert(cc > 0)
			bypass = F.pad(bypass, (-c, -c, -c, -c, -cc, -cc))
		return torch.cat((upsampled, bypass), 1)
	
	def forward(self, input):
		conv1 = F.leaky_relu(self.conv1(input), 0.005)
		conv2 = F.leaky_relu(self.conv2(conv1), 0.005)
		pool1 = self.pool1(conv2)
		conv3 = F.leaky_relu(self.conv3(pool1), 0.005)
		conv4 = F.leaky_relu(self.conv4(conv3), 0.005)
		pool2 = self.pool2(conv4)
		conv5 = F.leaky_relu(self.conv5(pool2), 0.005)
		conv6 = F.leaky_relu(self.conv6(conv5), 0.005)
		pool3 = self.pool3(conv6)
		conv7 = F.leaky_relu(self.conv7(pool3), 0.005)
		conv8 = F.leaky_relu(self.conv8(conv7), 0.005)

		dconv1 = self.dconv1(conv8)
		conv9 = self.conv9(dconv1)
		mc1 = self.crop_and_concat(conv9, conv6, crop=True)
		conv10 = F.leaky_relu(self.conv10(mc1), 0.005)
		conv11 = F.leaky_relu(self.conv11(conv10), 0.005)

		dconv2 = self.dconv2(conv11)
		conv12 = self.conv12(dconv2)
		mc2 = self.crop_and_concat(conv12, conv4, crop=True)
		conv13 = F.leaky_relu(self.conv13(mc2), 0.005)
		conv14 = F.leaky_relu(self.conv14(conv13), 0.005)

		dconv3 = self.dconv3(conv14)
		conv15 = self.conv15(dconv3)
		mc3 = self.crop_and_concat(conv15, conv2, crop=True)
		conv16 = F.leaky_relu(self.conv16(mc3), 0.005)
		conv17 = F.leaky_relu(self.conv17(conv16), 0.005)
		output = self.conv18(conv17)
		# if self.if_sigmoid:
		# 	output = torch.sigmoid(output)
		# if self.show_feature:
		# 	return conv8, conv11, conv14, conv17, output
		# else:
		# 	return output
		return output

class UNet3D_MALA_embedding_small6(nn.Module):
	def __init__(self, output_nc=3, if_sigmoid=False, init_mode='kaiming', show_feature=False, emd=16):
		super(UNet3D_MALA_embedding_small6, self).__init__()
		self.if_sigmoid = if_sigmoid
		self.init_mode = init_mode
		self.show_feature = show_feature
		self.emd = emd
		self.conv1 = nn.Conv3d(1, 4, 3, stride=1, padding=0, dilation=1, groups=1, bias=True) #8-4
		self.conv2 = nn.Conv3d(4, 4, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

		self.conv3 = nn.Conv3d(4, 8, 3, stride=1, padding=0, dilation=1, groups=1, bias=True) #16-8
		self.conv4 = nn.Conv3d(8, 8, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

		self.conv5 = nn.Conv3d(8, 16, 3, stride=1, padding=0, dilation=1, groups=1, bias=True) #32-16
		self.conv6 = nn.Conv3d(16, 16, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.pool3 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))

		self.conv7 = nn.Conv3d(16, 32, 3, stride=1, padding=0, dilation=1, groups=1, bias=True) #64-32
		self.conv8 = nn.Conv3d(32, 32, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.dconv1 = nn.ConvTranspose3d(32, 32, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=32,
										 bias=False)
		self.conv9 = nn.Conv3d(32, 16, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv10 = nn.Conv3d(32, 16, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv11 = nn.Conv3d(16, 16, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.dconv2 = nn.ConvTranspose3d(16, 16, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=16,
										 bias=False)
		self.conv12 = nn.Conv3d(16, 8, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv13 = nn.Conv3d(16, 16, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv14 = nn.Conv3d(16, 16, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.dconv3 = nn.ConvTranspose3d(16, 16, (1, 3, 3), stride=(1, 3, 3), padding=0, dilation=1, groups=16,
										 bias=False)
		self.conv15 = nn.Conv3d(16, 4, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv16 = nn.Conv3d(8, 4, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
		self.conv17 = nn.Conv3d(4, 4, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.conv18 = nn.Conv3d(4, self.emd, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
		# self.apply(self._weight_init)
		# Initialization
		for m in self.modules():
			if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
				if self.init_mode == 'kaiming':
					init.kaiming_normal_(m.weight, 0.005, 'fan_in', 'leaky_relu')
				elif self.init_mode == 'xavier':
					init.xavier_normal_(m.weight)
				elif self.init_mode == 'orthogonal':
					init.orthogonal_(m.weight)
				else:
					raise AttributeError('No this init mode!')

	@staticmethod
	def _weight_init(m):
		if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
			init.kaiming_normal_(m.weight, 0.005, 'fan_in', 'leaky_relu')

	def crop_and_concat(self, upsampled, bypass, crop=False):
		if crop:
			c = (bypass.size()[3] - upsampled.size()[3]) // 2
			cc = (bypass.size()[2] - upsampled.size()[2]) // 2
			assert (c > 0)
			assert (cc > 0)
			bypass = F.pad(bypass, (-c, -c, -c, -c, -cc, -cc))
		return torch.cat((upsampled, bypass), 1)

	def forward(self, input):
		conv1 = F.leaky_relu(self.conv1(input), 0.005)
		conv2 = F.leaky_relu(self.conv2(conv1), 0.005)
		pool1 = self.pool1(conv2)
		conv3 = F.leaky_relu(self.conv3(pool1), 0.005)
		conv4 = F.leaky_relu(self.conv4(conv3), 0.005)
		pool2 = self.pool2(conv4)
		conv5 = F.leaky_relu(self.conv5(pool2), 0.005)
		conv6 = F.leaky_relu(self.conv6(conv5), 0.005)
		pool3 = self.pool3(conv6)
		conv7 = F.leaky_relu(self.conv7(pool3), 0.005)
		conv8 = F.leaky_relu(self.conv8(conv7), 0.005)
		dconv1 = self.dconv1(conv8)
		conv9 = self.conv9(dconv1)
		mc1 = self.crop_and_concat(conv9, conv6, crop=True)
		conv10 = F.leaky_relu(self.conv10(mc1), 0.005)
		conv11 = F.leaky_relu(self.conv11(conv10), 0.005)
		dconv2 = self.dconv2(conv11)
		conv12 = self.conv12(dconv2)
		mc2 = self.crop_and_concat(conv12, conv4, crop=True)
		conv13 = F.leaky_relu(self.conv13(mc2), 0.005)
		conv14 = F.leaky_relu(self.conv14(conv13), 0.005)
		dconv3 = self.dconv3(conv14)
		conv15 = self.conv15(dconv3)
		mc3 = self.crop_and_concat(conv15, conv2, crop=True)
		conv16 = F.leaky_relu(self.conv16(mc3), 0.005)
		conv17 = F.leaky_relu(self.conv17(conv16), 0.005)
		output = self.conv18(conv17)
		# if self.if_sigmoid:
		# 	output = torch.sigmoid(output)
		if self.show_feature:
			return conv8, conv11, conv14, conv17, output
		# else:
		# 	return output
		return output

if __name__ == '__main__':
	""" example of weight sharing """
	#self.convs1_siamese = Conv3x3Stack(1, 12, negative_slope)
	#self.convs1_siamese[0].weight = self.convs1[0].weight
	
	import numpy as np
	# from model.model_para import model_structure
	from ptflops import get_model_complexity_info
	model = UNet3D_MALA_embedding(if_sigmoid=True, init_mode='kaiming').cuda()
	# model_structure(model)
	macs, params = get_model_complexity_info(model, (1,  84, 268, 268), as_strings=True,
											 print_per_layer_stat=True, verbose=True)

	x = torch.tensor(np.random.random((1, 1, 84, 268, 268)).astype(np.float32)).cuda()
	out = model(x)
	print(out.shape) # (1, 3, 56, 56, 56)

	print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
	print('{:<30}  {:<8}'.format('Number of parameters: ', params))
