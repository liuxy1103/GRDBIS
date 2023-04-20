import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import PIL
import torchvision
from torchvision.models import resnet18
import torchvision.transforms as transforms

print(torch.__version__)
from sklearn.metrics import accuracy_score
import torch.optim as optim
import tqdm

torch.manual_seed(1)
np.random.seed(1)


class VAE(nn.Module):
    def __init__(self, image_channels=16,image_size=544):
        super(VAE, self).__init__()
        # 定义编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels,32,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(256),
            # nn.LeakyReLU(0.2, inplace=True),
        )
        self.hidden_dimension = int(128*(image_size/8)*(image_size/8))
        self.z_dimension = 16
        # self.encoder_fc1=nn.Linear(self.hidden_dimension,self.z_dimension)
        # self.encoder_fc2=nn.Linear(self.hidden_dimension,self.z_dimension)
        self.encoder1 =  nn.Conv2d(128,  self.z_dimension, kernel_size=1, stride=1, padding=0)
        self.encoder2 = nn.Conv2d(128, self.z_dimension, kernel_size=1, stride=1, padding=0)

        self.decoder3 = nn.Conv2d(self.z_dimension,128, kernel_size=1, stride=1, padding=0)
        self.Sigmoid = nn.Sigmoid()
        self.decoder_fc = nn.Linear(self.z_dimension,self.hidden_dimension)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(inplace=True),
        )

    def noise_reparameterize(self,mean,logvar):
        eps = torch.randn(mean.shape).cuda()
        z = mean + eps * torch.exp(logvar)
        return z

    def forward(self, x):
        self.ori_shape = x.shape
        out1,out2 = self.encoder(x),self.encoder(x)
        # mean = self.encoder_fc1(out1.view(out1.shape[0],-1))
        # logstd = self.encoder_fc2(out2.view(out2.shape[0],-1))
        mean = self.encoder1(out1)
        logstd = self.encoder2(out2)
        z = self.noise_reparameterize(mean,logstd)  # z, mu, logvar
        out3 = self.decoder3(z)
        # out3 = out3.view(out3.shape[0],out1.shape[-3],out1.shape[-2],out1.shape[-1])
        out3 = self.decoder(out3)
        return out3,mean,logstd



class VAE2(nn.Module):
    def __init__(self, image_channels=16,image_size=544):
        super(VAE2, self).__init__()
        # 定义编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels,32,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.z_dimension = 16
        self.hidden_dimension = int(128*(image_size/32)*(image_size/32))
        self.encoder1 =  nn.Conv2d(512,  self.z_dimension, kernel_size=1, stride=1, padding=0)
        self.encoder2 = nn.Conv2d(512, self.z_dimension, kernel_size=1, stride=1, padding=0)

        self.decoder3 = nn.Conv2d(self.z_dimension,512, kernel_size=1, stride=1, padding=0)
        self.Sigmoid = nn.Sigmoid()
        self.decoder_fc = nn.Linear(self.z_dimension,self.hidden_dimension)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(inplace=True),
        )

    def noise_reparameterize(self,mean,logvar):
        eps = torch.randn(mean.shape).cuda()
        z = mean + eps * torch.exp(logvar)
        return z

    def forward(self, x):
        self.ori_shape = x.shape
        out1,out2 = self.encoder(x),self.encoder(x)
        # mean = self.encoder_fc1(out1.view(out1.shape[0],-1))
        # logstd = self.encoder_fc2(out2.view(out2.shape[0],-1))
        mean = self.encoder1(out1)
        logstd = self.encoder2(out2)
        z = self.noise_reparameterize(mean,logstd)  # z, mu, logvar
        out3 = self.decoder3(z)
        # out3 = out3.view(out3.shape[0],out1.shape[-3],out1.shape[-2],out1.shape[-1])
        out3 = self.decoder(out3)
        return out3,mean,logstd

class VAE3(nn.Module):
    def __init__(self, image_channels=16,image_size=544):
        super(VAE3, self).__init__()
        # 定义编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels,32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),
        )
        self.z_dimension = 16
        self.encoder1 =  nn.Conv2d(128,  self.z_dimension, kernel_size=1, stride=1, padding=0)
        self.encoder2 = nn.Conv2d(128, self.z_dimension, kernel_size=1, stride=1, padding=0)
        self.decoder3 = nn.Conv2d(self.z_dimension,128, kernel_size=1, stride=1, padding=0)
        self.Sigmoid = nn.Sigmoid()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

    def noise_reparameterize(self,mean,logvar):
        eps = torch.randn(mean.shape).cuda()
        z = mean + eps * torch.exp(logvar)
        return z

    def forward(self, x):
        self.ori_shape = x.shape
        out1,out2 = self.encoder(x),self.encoder(x)
        # mean = self.encoder_fc1(out1.view(out1.shape[0],-1))
        # logstd = self.encoder_fc2(out2.view(out2.shape[0],-1))
        mean = self.encoder1(out1)
        logstd = self.encoder2(out2)
        z = self.noise_reparameterize(mean,logstd)  # z, mu, logvar
        out3 = self.decoder3(z)
        # out3 = out3.view(out3.shape[0],out1.shape[-3],out1.shape[-2],out1.shape[-1])
        out3 = self.decoder(out3)
        return out3,mean,logstd

class VAE4(nn.Module):
    def __init__(self, image_channels=16,image_size=544):
        super(VAE4, self).__init__()
        # 定义编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels,32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.z_dimension = 16
        self.encoder1 =  nn.Conv2d(256,  self.z_dimension, kernel_size=1, stride=1, padding=0)  # out:hxwx16
        self.encoder2 = nn.Conv2d(256, self.z_dimension, kernel_size=1, stride=1, padding=0)
        self.decoder3 = nn.Conv2d(self.z_dimension,256, kernel_size=1, stride=1, padding=0)
        self.Sigmoid = nn.Sigmoid()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

    def noise_reparameterize(self,mean,logvar):
        eps = torch.randn(mean.shape).cuda()
        z = mean + eps * torch.exp(logvar)
        return z

    def forward(self, x):
        self.ori_shape = x.shape
        out1,out2 = self.encoder(x),self.encoder(x)
        # mean = self.encoder_fc1(out1.view(out1.shape[0],-1))
        # logstd = self.encoder_fc2(out2.view(out2.shape[0],-1))
        mean = self.encoder1(out1)
        logstd = self.encoder2(out2)
        z = self.noise_reparameterize(mean,logstd)  # z, mu, logvar
        out3 = self.decoder3(z)
        # out3 = out3.view(out3.shape[0],out1.shape[-3],out1.shape[-2],out1.shape[-1])
        out3 = self.decoder(out3)
        return out3,mean,logstd


class VAE5(nn.Module):
    def __init__(self, image_channels=16,image_size=544):
        super(VAE5, self).__init__()
        # 定义编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels,image_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(image_channels),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(image_channels,image_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(image_channels),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(image_channels,image_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(image_channels),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(image_channels, image_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(image_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.z_dimension = 2
        self.encoder1 =  nn.Conv2d(image_channels,  self.z_dimension, kernel_size=1, stride=1, padding=0)  # out:hxwx16
        self.encoder2 = nn.Conv2d(image_channels, self.z_dimension, kernel_size=1, stride=1, padding=0)
        self.decoder3 = nn.Conv2d(self.z_dimension,image_channels, kernel_size=1, stride=1, padding=0)
        self.Sigmoid = nn.Sigmoid()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(image_channels, image_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(image_channels, image_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(image_channels, image_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(image_channels, image_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

    def noise_reparameterize(self,mean,logvar):
        eps = torch.randn(mean.shape).cuda()
        z = mean + eps * torch.exp(logvar)
        return z

    def forward(self, x):
        self.ori_shape = x.shape
        out1,out2 = self.encoder(x),self.encoder(x)
        # mean = self.encoder_fc1(out1.view(out1.shape[0],-1))
        # logstd = self.encoder_fc2(out2.view(out2.shape[0],-1))
        mean = self.encoder1(out1)
        logstd = self.encoder2(out2)
        z = self.noise_reparameterize(mean,logstd)  # z, mu, logvar
        out3 = self.decoder3(z)
        # out3 = out3.view(out3.shape[0],out1.shape[-3],out1.shape[-2],out1.shape[-1])
        out3 = self.decoder(out3)
        return out3,mean,logstd

def vae_loss_function(recon_x,x,mean,std):
    # BCE = F.binary_cross_entropy(recon_x,x,reduction='sum')
    l2 = F.mse_loss(recon_x, x, reduction='sum')
    l1 = F.l1_loss(recon_x, x, reduction='sum')
    # 因为var是标准差的自然对数，先求自然对数然后平方转换成方差
    var = torch.pow(torch.exp(std),2)
    KLD = -0.5 * torch.sum(1+torch.log(var+1e-8)-torch.pow(mean,2)-var)
    # return BCE+KLD

    # l2 = F.mse_loss(recon_x, x, reduction='sum')
    # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # l1 = F.l1_loss(recon_x, x, reduction='sum')
    return l1 + l2 + KLD, l2, l1, KLD

#loss,l2, l1, KLD = loss_function(x,img,mean,std)

if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    model_vae = VAE3().cuda()
    img = torch.randn((2,16,544,544)).cuda()
    out3,mean,logstd = model_vae(img)
    loss, l2, l1, KLD = vae_loss_function(out3, img, mean, logstd)
    print(loss,out3.shape)
    macs, params = get_model_complexity_info(model_vae, (16,544,544), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

