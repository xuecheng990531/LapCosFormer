import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
# from .VIT import *
# from .LapCos_Block import LapCos_Block
# from .decoder import *
# from .Semantic_Difference_Blocks import SDN
# from torchvision import models

from VIT import *
from LapCos_Block import LapCos_Block
from decoder import *
from Semantic_Difference_Blocks import SDN
from torchvision import models
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        width = int(out_channels * (base_width / 64))
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=2, groups=1, padding=1, dilation=1, bias=False)
        self.norm2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x_down = self.downsample(x)

        x = self.conv1(x)
        x = self.norm1(x)   
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)#torch.Size([1, 256, 64, 64])
        x = x + x_down
        x = self.relu(x)

        return x

class Encoder(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim):
        super().__init__()
        self.resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        
        # Redefine the first convolution to accept additional channel
        self.resnet.conv1 = nn.Conv2d(in_channels + 1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.c1=nn.Sequential(nn.Conv2d(64,128,1,1),nn.BatchNorm2d(128),nn.ReLU(inplace=True))
        self.c2=nn.Sequential(nn.Conv2d(256,256,3,2,1),nn.BatchNorm2d(256),nn.ReLU(inplace=True))
        self.c3=nn.Sequential(nn.Conv2d(512,1024,3,4,1),nn.BatchNorm2d(1024),nn.ReLU(inplace=True))
        self.c4=nn.Sequential(nn.Conv2d(512,512,3,2,1),nn.BatchNorm2d(512),nn.ReLU(inplace=True))
        # Initial layers without maxpool to control x1_0 output
        self.initial_layers = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu
        )
        
        # Use layer1 directly without adjusting channels
        self.layer1_blocks = self.resnet.layer1
        
        # Define remaining layers
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        
        # Convolution after layer1_blocks instead of before
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        
        # Vision Transformer (ViT) setup
        self.vit_img_dim = img_dim // patch_dim
        self.vit = ViT_tx(self.vit_img_dim, out_channels * 8, out_channels * 8,
                       head_num, mlp_dim, block_num, patch_dim=1, classification=False)

        # Convolutional block after ViT
        self.conv2 = nn.Conv2d(out_channels * 8, 512, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.lapcos = LapCos_Block(in_channels=1024, key_channels=1024, value_channels=1024, pyramid_levels=3)

    def forward(self, x, trimap):
        # Concatenate trimap with input image
        inp = torch.cat([x, trimap], dim=1)
        
        # Pass through initial layers
        x1_0 = self.initial_layers(inp)
        x1_a = self.c1(x1_0)#1,128,256，256

        x1 = self.layer1_blocks(x1_0)
        x1_b = self.c2(x1)#1,256,128,128
        
        x2 = self.layer2(x1)#torch.Size([1, 512, 128, 128])
        x2_c= self.c4(x2)#torch.Size([1, 512, 64, 64])
        x2 = self.c3(x2)
        
        x = self.lapcos(x2)
        
        
        trimap_resized = F.interpolate(trimap, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        trimap_final = trimap_resized.expand(-1, 1024, -1, -1)

        x = self.vit(x, trimap_final)
        x = rearrange(x, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim) #1,1024,32,32


        x = self.conv2(x)
        x = self.norm2(x)
        final = self.relu(x)#1，512，32，32

        return final, x1_a, x1_b,x2_c






class Decoder(nn.Module):
    def __init__(self, out_channels, class_num):
        super().__init__()

        self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 2,ecb=False)
        self.decoder2 = DecoderBottleneck(out_channels * 4, out_channels,ecb=False)
        self.decoder3 = DecoderBottleneck(out_channels * 2, int(out_channels * 1 / 2),ecb=False)
        self.decoder4 = DecoderBottleneck(int(out_channels * 1 / 2), int(out_channels * 1 / 8),ecb=True)

        self.conv1 = nn.Conv2d(int(out_channels * 1 / 8), class_num, kernel_size=1)
        self.sdn=SDN(16,16)

    def forward(self, x, x1, x2, x3):
        # torch.Size([1, 512, 32, 32]) torch.Size([1, 128, 128, 128]) torch.Size([1, 256, 64, 64]) torch.Size([1, 512, 32, 32])     
        x = self.decoder1(x, x3)#torch.Size([1, 512, 16, 16])
        x = self.decoder2(x, x2)#torch.Size([1, 256, 32, 32])
        xa = self.decoder3(x, x1)#torch.Size([1, 64, 256, 256])
        x = self.decoder4(xa)#torch.Size([1, 16, 512, 512])
        sdn=self.sdn(xa,x)
        x = self.conv1(sdn)#16-1

        return x

class TransUNet(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim, class_num):
        super().__init__()
        self.encoder = Encoder(img_dim, in_channels, out_channels,
                               head_num, mlp_dim, block_num, patch_dim)

        self.decoder = Decoder(out_channels, class_num)

    def forward(self, x,y):
        x, x1, x2, x3 = self.encoder(x,y)
        x = self.decoder(x, x1, x2, x3)

        return F.sigmoid(x)
        # return x


if __name__ == '__main__':
    import torch

    transunet = TransUNet(img_dim=512,
                          in_channels=3,
                          out_channels=128,
                          head_num=4,
                          mlp_dim=256,
                          block_num=2,
                          patch_dim=16,
                          class_num=1).cuda()
    print(sum(p.numel() for p in transunet.parameters()))
    x=torch.randn(1, 3, 512, 512).cuda()
    y=torch.randn(1, 1, 512, 512).cuda()
    print(transunet(x,y).shape)