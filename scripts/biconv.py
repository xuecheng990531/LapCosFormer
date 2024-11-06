import torch
import math
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Optional
# from mobilenet import BinaryActivation, LearnableBias, BiConv2d

"""
Adopted from <https://github.com/wuhuikai/DeepGuidedFilter/>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

stage_out_channel = [16] + [32, 64, 32] + [64, 128, 64] * 2 + [128, 256, 128] * 2 + [256] + [1024]

class BiConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(BiConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, \
                                            padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.channel_threshold = nn.Parameter(torch.zeros((1, in_channels, 1, 1), requires_grad=True))
        
    def forward(self, input):
        weight = self.weight
        activation = input
        weight = weight - torch.mean(torch.mean(torch.mean(weight,dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(weight),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weight_no_grad = scaling_factor * torch.sign(weight)
        cliped_weight = torch.clamp(weight, -1.0, 1.0)
        binary_weight = binary_weight_no_grad.detach() - cliped_weight.detach() + cliped_weight
        output = F.conv2d(activation, binary_weight, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return BiConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return BiConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class firstconv3x3(nn.Module):
    def __init__(self, inp, oup, stride):
        super(firstconv3x3, self).__init__()

        self.conv1 = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(oup)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)

        return out

class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        cliped_out = torch.clamp(x, -1.0, 1.0)
        binary_out = out_forward.detach() - cliped_out.detach() + cliped_out
        out = out_forward.detach() - binary_out.detach() + binary_out
        return out

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out
    
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d

        self.move11 = LearnableBias(inplanes)
        self.binary_3x3= conv3x3(inplanes, inplanes, stride=stride)
        self.bn1 = norm_layer(inplanes)

        self.move12 = LearnableBias(inplanes)
        self.prelu1 = nn.PReLU(inplanes)
        self.move13 = LearnableBias(inplanes)

        self.move21 = LearnableBias(inplanes)

        if inplanes == planes:
            self.binary_pw = conv1x1(inplanes, planes)
            self.bn2 = norm_layer(planes)
        elif 2 * inplanes == planes:
            self.binary_pw_down1 = conv1x1(inplanes, inplanes)
            self.binary_pw_down2 = conv1x1(inplanes, inplanes)
            self.bn2_1 = norm_layer(inplanes)
            self.bn2_2 = norm_layer(inplanes)
        elif 4 * inplanes == planes:
            self.binary_pw_down1 = conv1x1(inplanes, inplanes)
            self.binary_pw_down2 = conv1x1(inplanes, inplanes)
            self.binary_pw_down3 = conv1x1(inplanes, inplanes)
            self.binary_pw_down4 = conv1x1(inplanes, inplanes)
            self.bn2_1 = norm_layer(inplanes)
            self.bn2_2 = norm_layer(inplanes)
            self.bn2_3 = norm_layer(inplanes)
            self.bn2_4 = norm_layer(inplanes)
        else:
            self.binary_pw_down1 = conv1x1(inplanes, planes)
            self.bn2_1 = norm_layer(planes)

        self.move22 = LearnableBias(planes)
        self.prelu2 = nn.PReLU(planes)
        self.move23 = LearnableBias(planes)

        self.binary_activation = BinaryActivation()
        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes

        if self.inplanes != self.planes:
            self.pooling = nn.AvgPool2d(2,2,ceil_mode=True)

    def forward(self, x):

        out1 = self.move11(x)

        out1 = self.binary_activation(out1)
        out1 = self.binary_3x3(out1)
        out1 = self.bn1(out1)

        if self.stride == 2:
            x = self.pooling(x)

        out1 = x + out1

        out1 = self.move12(out1)
        out1 = self.prelu1(out1)
        out1 = self.move13(out1)

        out2 = self.move21(out1)
        out2 = self.binary_activation(out2)

        if self.inplanes == self.planes:
            out2 = self.binary_pw(out2)
            out2 = self.bn2(out2)
            out2 += out1
        elif self.inplanes * 2 == self.planes:
            out2_1 = self.binary_pw_down1(out2)
            out2_2 = self.binary_pw_down2(out2)
            out2_1 = self.bn2_1(out2_1)
            out2_2 = self.bn2_2(out2_2)
            out2_1 += out1
            out2_2 += out1
            out2 = torch.cat([out2_1, out2_2], dim=1)
        elif self.planes == self.inplanes * 4:
            out2_1 = self.binary_pw_down1(out2)
            out2_2 = self.binary_pw_down2(out2)
            out2_3 = self.binary_pw_down3(out2)
            out2_4 = self.binary_pw_down4(out2)
            out2_1 = self.bn2_1(out2_1)
            out2_2 = self.bn2_2(out2_2)
            out2_3 = self.bn2_3(out2_3)
            out2_4 = self.bn2_4(out2_4)
            out2_1 += out1
            out2_2 += out1
            out2_3 += out1
            out2_4 += out1
            out2 = torch.cat([out2_1, out2_2, out2_3, out2_4], dim=1)
        else:
            assert self.planes == self.inplanes // 2
            split_result = torch.split(out1, self.planes, dim = 1)
            split_result = sum(split_result) / 2
            out2 = self.binary_pw_down1(out2)
            out2 = self.bn2_1(out2)
            out2 += split_result

        out2 = self.move22(out2)
        out2 = self.prelu2(out2)
        out2 = self.move23(out2)

        return out2

class BiMobileNet(nn.Module):
    def __init__(self, pretrained: bool = False):
        super(BiMobileNet, self).__init__()
        self.feature = nn.ModuleList()
        for i in range(len(stage_out_channel)):
            if i == 0:
                self.feature.append(firstconv3x3(3, stage_out_channel[i], 2))
            elif i in [2, 5, 11]:
                self.feature.append(BasicBlock(stage_out_channel[i-1], stage_out_channel[i], 2))
            else:
                self.feature.append(BasicBlock(stage_out_channel[i-1], stage_out_channel[i], 1))
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(stage_out_channel[-1], 1000)

        self.pooling = nn.AvgPool2d(2,2,ceil_mode=True)

        if pretrained:
            checkpoint = torch.load('pretrained_models/bimobilenet_best.pth.tar', map_location='cpu')
            self.load_state_dict({'.'.join(k.split('.')[1:]): checkpoint['state_dict'][k] for k in checkpoint['state_dict'].keys()})
        
        del self.pool1
        del self.fc

    def forward_single_frame(self, x):
        x = normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        blocks = [[1, 3], [4, 6], [7, 9], [10, 12], [13, 15], [-1, -1]]
        i_block = 0
        feature = None
        for i, block in enumerate(self.feature):
            if i == blocks[i_block][0]:
                x = block(x)
                feature = x
            elif i == blocks[i_block][1]:
                x = block(x)
                if feature.shape[2] != x.shape[2]:
                    x += self.pooling(feature)
                i_block += 1
            else:
                x = block(x)
            if i == 0:
                f1 = x
            elif i == 3:
                f2 = x
            elif i == 9:
                f3 = x
            elif i == 17:
                f4 = x
        return [f1, f2, f3, f4]
    
    def forward_time_series(self, x):
        B, T = x.shape[:2]
        features = self.forward_single_frame(x.flatten(0, 1))
        features = [f.unflatten(0, (B, T)) for f in features]
        return features

    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)



def get_incoherent_mask(input_masks, sfact):
    mask = input_masks.float()
    w = input_masks.shape[-1]
    h = input_masks.shape[-2]
    mask_small = F.interpolate(mask, (h//sfact, w//sfact), mode='bilinear')
    mask_recover = F.interpolate(mask_small, (h, w), mode='bilinear')
    mask_residue = (mask - mask_recover).abs()
    mask_residue = (mask_residue >= 0.01).float()
    return mask_residue


class BiRecurrentDecoder(nn.Module):
    # [16, 32, 64, 128], [80, 40, 32, 16]
    def __init__(self, feature_channels, decoder_channels):
        super().__init__()
        self.avgpool = AvgPool()
        self.decode4 = BottleneckBlock(feature_channels[3])
        self.decode3 = UpsamplingBlock(feature_channels[3], feature_channels[2], 3, decoder_channels[0], 3, 1)
        self.decode2 = UpsamplingBlock(decoder_channels[0], feature_channels[1], 3, decoder_channels[1], 3, 1) #3
        self.decode1 = UpsamplingBlock(decoder_channels[1], feature_channels[0], 3, decoder_channels[2], 3, 1) #3
        self.decode0 = OutputBlock(decoder_channels[2], 3, decoder_channels[3])

    def forward(self,
                s0: Tensor, f1: Tensor, f2: Tensor, f3: Tensor, f4: Tensor,
                r1: Optional[Tensor], r2: Optional[Tensor],
                r3: Optional[Tensor], r4: Optional[Tensor], project_mat):
        s1, s2, s3 = self.avgpool(s0)
        
        x4, r4, x4_out = self.decode4(f4, r4) # encoded feature at f4 level, r4
        

        _, pha_s = project_mat(x4_out).split([3, 1], dim=-3)
        pha_s = pha_s.clamp(0., 1.)
        # print('phas:', pha_s.shape)
        binary_mask = pha_s.gt(0)
        B, T, _, _, _ = binary_mask.shape

        binary_mask = get_incoherent_mask(binary_mask.flatten(0, 1), 2) 
        
        x3, r3 = self.decode3(x4, f3, s3, r3, binary_mask)
        
        x2, r2 = self.decode2(x3, f2, s2, r2, binary_mask)
        
        x1, r1 = self.decode1(x2, f1, s1, r1, binary_mask)

        x0 = self.decode0(x1, s0, binary_mask)

        return x0, x4_out, r1, r2, r3, r4    


class AvgPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AvgPool2d(2, 2, count_include_pad=False, ceil_mode=True)
        
    def forward_single_frame(self, s0):
        s1 = self.avgpool(s0)
        s2 = self.avgpool(s1)
        s3 = self.avgpool(s2)
        return s1, s2, s3
    
    def forward_time_series(self, s0):
        B, T = s0.shape[:2]
        s0 = s0.flatten(0, 1)
        s1, s2, s3 = self.forward_single_frame(s0)
        s1 = s1.unflatten(0, (B, T))
        s2 = s2.unflatten(0, (B, T))
        s3 = s3.unflatten(0, (B, T))
        return s1, s2, s3
    
    def forward(self, s0):
        if s0.ndim == 5:
            return self.forward_time_series(s0)
        else:
            return self.forward_single_frame(s0)


class BottleneckBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.gru = ConvGRU(channels // 2)
        self.conv = nn.Sequential(
            LearnableBias(channels),
            BinaryActivation(),
            BiConv2d(channels, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            LearnableBias(16),
            nn.PReLU(16),
        )
        
    def forward(self, x, r: Optional[Tensor]):
        a, b = x.split(self.channels // 2, dim=-3)
        b, r = self.gru(b, r)
        x = torch.cat([a, b], dim=-3)
        B, T, _, H, W = x.shape
        x_out = self.conv(x.flatten(0, 1)).unflatten(0, (B, T))
        return x, r, x_out

    
class UpsamplingBlock(nn.Module):
     # [16, 32, 64, 128], [80, 40, 32, 16]
    def __init__(self, in_channels, skip_channels, src_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.out_channels = out_channels
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            LearnableBias(in_channels + skip_channels + src_channels),
            BinaryActivation(),
            BiConv2d(in_channels + skip_channels + src_channels, out_channels, kernel_size, 1, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            LearnableBias(out_channels),
            nn.PReLU(out_channels),
        )
        self.conv_1x1 = nn.Sequential(
            LearnableBias(in_channels + skip_channels + src_channels),
            BinaryActivation(),
            BiConv2d(in_channels + skip_channels + src_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            LearnableBias(out_channels),
            nn.PReLU(out_channels),
        )
        self.gru = ConvGRU_Up(out_channels // 2)

    def forward_single_frame(self, x, f, s, r: Optional[Tensor], inc_mask):
        x = self.upsample(x)
        x = x[:, :, :s.size(2), :s.size(3)]
        x = torch.cat([x, f, s], dim=1)
        x = self.conv(x)
        # print('2-x shape:', x.shape, 'inc_mask shape:', inc_mask.shape)
        a, b = x.split(self.out_channels // 2, dim=1)
        b, r = self.gru(b, r)
        x = torch.cat([a, b], dim=1)
        return x, r
    
    def forward_time_series(self, x, f, s, r: Optional[Tensor], inc_mask):
        B, T, _, H, W = s.shape
        x = x.flatten(0, 1)
        f = f.flatten(0, 1)
        s = s.flatten(0, 1)
        x = self.upsample(x)
        x = x[:, :, :H, :W]
        x = torch.cat([x, f, s], dim=1)
        x_1 = self.conv_1x1(x)
        x = self.conv(x)
        inc_mask = F.interpolate(inc_mask, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x_new = x * inc_mask + x_1
        
        x_new = x_new.unflatten(0, (B, T))
        a, b = x_new.split(self.out_channels // 2, dim=2)
        b, r = self.gru(b, r, inc_mask)
        x_new = torch.cat([a, b], dim=2)
        return x_new, r
    
    def forward(self, x, f, s, r: Optional[Tensor], inc_mask):
        if x.ndim == 5:
            return self.forward_time_series(x, f, s, r, inc_mask)
        else:
            return self.forward_single_frame(x, f, s, r, inc_mask)


class OutputBlock(nn.Module):
    def __init__(self, in_channels, src_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            LearnableBias(in_channels + src_channels),
            BinaryActivation(),
            BiConv2d(in_channels + src_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            # LearnableBias(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            # LearnableBias(out_channels),
            nn.ReLU(True),
        )

        self.conv_1x1 = nn.Sequential(
            LearnableBias(in_channels + src_channels),
            BinaryActivation(),
            BiConv2d(in_channels + src_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            # LearnableBias(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            # LearnableBias(out_channels),
            nn.ReLU(True),
        )
        
    def forward_single_frame(self, x, s):
        x = self.upsample(x)
        x = x[:, :, :s.size(2), :s.size(3)]
        x = torch.cat([x, s], dim=1)
        x = self.conv(x)
        return x
    
    def forward_time_series(self, x, s, inc_mask):
        B, T, _, H, W = s.shape
        x = x.flatten(0, 1)
        s = s.flatten(0, 1)
        x = self.upsample(x)
        x = x[:, :, :H, :W]
        x = torch.cat([x, s], dim=1)
        x_1 = self.conv_1x1(x)
        x = self.conv(x)
        inc_mask = F.interpolate(inc_mask, size=x.shape[-2:], mode='bilinear', align_corners=False)
        # print('inc_mask shape:', inc_mask.shape, 'x shape:', x.shape)
        x_new = x * inc_mask + x_1
        #print('x shape in output:', x.shape)
        x_new = x_new.unflatten(0, (B, T))
        return x_new
    
    def forward(self, x, s, inc_mask):
        if x.ndim == 5:
            return self.forward_time_series(x, s, inc_mask)
        else:
            return self.forward_single_frame(x, s)


class ConvGRU(nn.Module):
    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 padding: int = 1):
        super().__init__()
        self.channels = channels
        self.ih = nn.Sequential(
            LearnableBias(channels * 2),
            BinaryActivation(),
            BiConv2d(channels * 2, channels * 2, kernel_size, padding=padding),
            # LearnableBias(channels * 2),
            nn.Sigmoid()
        )
        self.hh = nn.Sequential(
            LearnableBias(channels * 2),
            BinaryActivation(),
            BiConv2d(channels * 2, channels, kernel_size, padding=padding),
            # LearnableBias(channels),
            nn.Tanh()
        )
        
    def forward_single_frame(self, x, h):
        r, z = self.ih(torch.cat([x, h], dim=1)).split(self.channels, dim=1)
        c = self.hh(torch.cat([x, r * h], dim=1))
        h = (1 - z) * h + z * c
        return h, h
    
    def forward_time_series(self, x, h):
        o = []
        for xt in x.unbind(dim=1):
            ot, h = self.forward_single_frame(xt, h)
            o.append(ot)
        o = torch.stack(o, dim=1)
        return o, h
        
    def forward(self, x, h: Optional[Tensor]):
        if h is None:
            h = torch.zeros((x.size(0), x.size(-3), x.size(-2), x.size(-1)),
                            device=x.device, dtype=x.dtype)
        
        if x.ndim == 5:
            return self.forward_time_series(x, h)
        else:
            return self.forward_single_frame(x, h)


class ConvGRU_Up(nn.Module):
    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 padding: int = 1):
        super().__init__()
        self.channels = channels
        self.ih = nn.Sequential(
            LearnableBias(channels * 2),
            BinaryActivation(),
            BiConv2d(channels * 2, channels * 2, kernel_size, padding=padding),
            LearnableBias(channels * 2),
            nn.Sigmoid()
        )
        self.ih_1x1 = nn.Sequential(
            LearnableBias(channels * 2),
            BinaryActivation(),
            BiConv2d(channels * 2, channels * 2, 1, padding=0),
            LearnableBias(channels * 2),
            nn.Sigmoid()
        )
        self.hh = nn.Sequential(
            LearnableBias(channels * 2),
            BinaryActivation(),
            BiConv2d(channels * 2, channels, kernel_size, padding=padding),
            LearnableBias(channels),
            nn.Tanh()
        )
        self.hh_1x1 = nn.Sequential(
            LearnableBias(channels * 2),
            BinaryActivation(),
            BiConv2d(channels * 2, channels, 1, padding=0),
            LearnableBias(channels),
            nn.Tanh()
        )
    
    def forward_single_frame(self, x, h, inc_mask_t):
        # print('x t shape:', x.shape, 'h t shape:', h.shape, 'inc_mask_t.shape:', inc_mask_t.shape)
        r, z = self.ih(torch.cat([x, h], dim=1)).split(self.channels, dim=1)
        r1, z1 = self.ih_1x1(torch.cat([x, h], dim=1)).split(self.channels, dim=1)
        r = r * inc_mask_t + r1 
        z = z * inc_mask_t + z1
        # print('r shape:', r.shape, 'z shape:', z.shape)
        c = self.hh(torch.cat([x, r * h], dim=1))
        c1 = self.hh_1x1(torch.cat([x, r * h], dim=1))
        c = c * inc_mask_t + c1
        h = (1 - z) * h + z * c
        return h, h
    
    def forward_time_series(self, x, h, inc_mask):
        o = []
        # print('x shape:', x.shape, 'inc_mask shape:', inc_mask.shape)
        for xt, inc_mask_t in zip(x.unbind(dim=1), inc_mask.unbind(dim=0)):
            ot, h = self.forward_single_frame(xt, h, inc_mask_t.unsqueeze(1))
            o.append(ot)
        o = torch.stack(o, dim=1)
        return o, h
    
    def forward(self, x, h: Optional[Tensor], inc_mask):
        if h is None:
            h = torch.zeros((x.size(0), x.size(-3), x.size(-2), x.size(-1)),
                            device=x.device, dtype=x.dtype)
        
        if x.ndim == 5:
            return self.forward_time_series(x, h, inc_mask)
        else:
            return self.forward_single_frame(x, h)      
    

class Projection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward_single_frame(self, x): 
        return self.conv(x)
    
    def forward_time_series(self, x):
        B, T = x.shape[:2]
        return self.conv(x.flatten(0, 1)).unflatten(0, (B, T))
        
    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)
    