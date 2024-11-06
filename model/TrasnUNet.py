import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from .VIT import ViT
from .qco_non_linear_quant import PTFEM
from .decoder import *

# from VIT import *
# from scripts.Trip_CrossAtt import CrossAttentionModule
# from qco_non_linear_quant import PTFEM
# from scripts.biconv import UpsamplingBlock,get_incoherent_mask,LearnableBias,BinaryActivation,BiConv2d
# from decoder import *


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
        self.conv1 = nn.Conv2d(in_channels+1, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)

        self.vit_img_dim = img_dim // patch_dim
        self.vit = ViT(self.vit_img_dim, out_channels * 8, out_channels * 8,
                       head_num, mlp_dim, block_num, patch_dim=1, classification=False)

        self.conv2 = nn.Conv2d(out_channels * 8, 512, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(512)
        self.ptfem=PTFEM()

    def forward(self, x,trimap):
        inp=torch.cat([x,trimap],dim=1)
        x = self.conv1(inp)
        x = self.norm1(x)
        x1 = self.relu(x)#torch.Size([1, 128, 128, 128])

        x2 = self.encoder1(x1)

        x3 = self.encoder2(x2)#torch.Size([1, 512, 32, 32])
        x = self.encoder3(x3)#torch.Size([1, 1024, 16, 16])

        x = self.vit(x)#torch.Size([1, 256, 1024])
        x = rearrange(x, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)#torch.Size([1, 1024, 16, 16])

        x = self.conv2(x)
        x = self.norm2(x)
        final = self.relu(x)#torch.Size([1, 512, 16, 16])


        # torch.Size([1, 512, 16, 16]) torch.Size([1, 128, 128, 128]) torch.Size([1, 256, 64, 64]) torch.Size([1, 512, 32, 32])
        return final, x1, x2, x3

class Decoder(nn.Module):
    def __init__(self, out_channels, class_num):
        super().__init__()

        self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 2,ecb=False)
        self.decoder2 = DecoderBottleneck(out_channels * 4, out_channels,ecb=False)
        self.decoder3 = DecoderBottleneck(out_channels * 2, int(out_channels * 1 / 2),ecb=False)
        self.decoder4 = DecoderBottleneck(int(out_channels * 1 / 2), int(out_channels * 1 / 8),ecb=True)

        self.conv1 = nn.Conv2d(int(out_channels * 1 / 8), class_num, kernel_size=1)


    def forward(self, x, x1, x2, x3):
        # torch.Size([1, 512, 8, 8]) torch.Size([1, 128, 64, 64]) torch.Size([1, 256, 32, 32]) torch.Size([1, 512, 16, 16])
        x = self.decoder1(x, x3)#torch.Size([1, 512, 16, 16])
        x = self.decoder2(x, x2)#torch.Size([1, 256, 32, 32])
        x = self.decoder3(x, x1)#256
        x = self.decoder4(x)#64-16
        x = self.conv1(x)#16-1

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

    transunet = TransUNet(img_dim=256,
                          in_channels=3,
                          out_channels=128,
                          head_num=4,
                          mlp_dim=512,
                          block_num=4,
                          patch_dim=16,
                          class_num=1).cuda()
    print(sum(p.numel() for p in transunet.parameters()))
    x=torch.randn(1, 3, 256, 256).cuda()
    y=torch.randn(1, 1, 256, 256).cuda()
    print(transunet(x,y).shape)