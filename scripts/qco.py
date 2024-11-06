
from numpy.core.fromnumeric import size
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules import module


class ConvBNReLU(nn.Module):
    '''Module for the Conv-BN-ReLU tuple.'''

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation=1, group=1,
                    has_bn=True, has_relu=True, mode='2d'):
        super(ConvBNReLU, self).__init__()
        self.has_bn = has_bn
        self.has_relu = has_relu
        self.conv = nn.Conv2d(
                c_in, c_out, kernel_size=kernel_size, stride=stride, 
                padding=padding, dilation=dilation, bias=False, groups=group)
        norm_layer = nn.BatchNorm2d
        if self.has_bn:
            self.bn = norm_layer(c_out)
        if self.has_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x



class QCO_2d(nn.Module):
    def __init__(self, scale, level_num):
        super(QCO_2d, self).__init__()
        self.f1 = nn.Sequential(ConvBNReLU(3, 64, 1, 1, 0, has_bn=False, has_relu=False, mode='2d'), nn.LeakyReLU(inplace=True))
        self.f2 = ConvBNReLU(64, 128, 1, 1, 0, has_bn=False, mode='2d')
        self.out = nn.Sequential(ConvBNReLU(256+128, 128, 1, 1, 0, has_bn=True, has_relu=True, mode='2d'), ConvBNReLU(128, 128, 1, 1, 0, has_bn=True, has_relu=False, mode='2d'))
        self.scale = scale
        self.level_num = level_num
    def forward(self, x):
        N1, C1, H1, W1 = x.shape
        if H1 // self.level_num != 0 or W1 // self.level_num != 0:
            x = F.adaptive_avg_pool2d(x, ((int(H1/self.level_num)*self.level_num), int(W1/self.level_num)*self.level_num))
        N, C, H, W = x.shape
        self.size_h = int(H / self.scale)
        self.size_w = int(W / self.scale)
        x_ave = F.adaptive_avg_pool2d(x, (self.scale, self.scale)) 
        x_ave_up = F.adaptive_avg_pool2d(x_ave, (H, W)) 
        cos_sim = (F.normalize(x_ave_up, dim=1) * F.normalize(x, dim=1)).sum(1) 
        cos_sim = cos_sim.unsqueeze(1) 
        cos_sim = cos_sim.reshape(N, 1, self.scale, self.size_h, self.scale, self.size_w)
        cos_sim = cos_sim.permute(0, 1, 2, 4, 3, 5)
        cos_sim = cos_sim.reshape(N, 1, int(self.scale*self.scale), int(self.size_h*self.size_w)) 
        cos_sim = cos_sim.permute(0, 1, 3, 2) 
        cos_sim = cos_sim.squeeze(1) 
        cos_sim_min, _ = cos_sim.min(1) 
        cos_sim_min = cos_sim_min.unsqueeze(-1) 
        cos_sim_max, _ = cos_sim.max(1) 
        cos_sim_max = cos_sim_max.unsqueeze(-1) 
        q_levels = torch.arange(self.level_num).float().cuda() 
        q_levels = q_levels.expand(N, self.scale*self.scale, self.level_num) 
        q_levels =  (2 * q_levels + 1) / (2 * self.level_num) * (cos_sim_max - cos_sim_min) + cos_sim_min 
        q_levels_inter = q_levels[:, :, 1] - q_levels[:, :, 0] 
        q_levels_inter = q_levels_inter.unsqueeze(1).unsqueeze(-1) 
        cos_sim = cos_sim.unsqueeze(-1) 
        q_levels = q_levels.unsqueeze(1)
        # quant = 1 - torch.abs(q_levels - cos_sim) 
        '''相比于硬量化，可以采用软量化，即对量化级别进行加权插值，而不是直接选择最近的离散值。
        比如，使用权重（如三角函数或高斯分布）在邻近的量化级别之间平滑过渡，以减少量化误差。
        可以通过基于相似度的加权平均来平滑不同量化级别的影响：'''
        # 降低量化误差
        quant = torch.softmax(-torch.abs(q_levels - cos_sim), dim=-1)


        quant = quant * (quant > (1 - q_levels_inter)) 
        quant = quant.view([N, self.size_h, self.size_w, self.scale*self.scale, self.level_num]) 
        quant = quant.permute(0, -2, -1, 1, 2) 
        quant = quant.reshape(N, -1, self.size_h, self.size_w)
        quant = F.pad(quant, (0, 1, 0, 1), mode='constant', value=0.)
        quant = quant.view(N, self.scale*self.scale, self.level_num, self.size_h+1, self.size_w+1)
        quant_left = quant[:, :, :, :self.size_h, :self.size_w].unsqueeze(3) 
        quant_right = quant[:, :, :, 1:, 1:].unsqueeze(2) 
        quant = quant_left * quant_right 
        sta = quant.sum(-1).sum(-1) 
        sta = sta / (sta.sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1) + 1e-6) 
        sta = sta.unsqueeze(1) 
        q_levels = q_levels.expand(self.level_num, N, 1, self.scale*self.scale, self.level_num)
        q_levels_h = q_levels.permute(1, 2, 3, 0, 4) 
        q_levels_w = q_levels_h.permute(0, 1, 2, 4, 3) 
        sta = torch.cat([q_levels_h, q_levels_w, sta], dim=1) 
        sta = sta.view(N, 3, self.scale * self.scale, -1) 

        sta = self.f1(sta)
        sta = self.f2(sta) 
        x_ave = x_ave.view(N, C, -1) 
        x_ave = x_ave.expand(self.level_num*self.level_num, N, C, self.scale*self.scale)
        x_ave = x_ave.permute(1, 2, 3, 0) 
        sta = torch.cat([x_ave, sta], dim=1)
        sta = self.out(sta) 
        sta = sta.mean(-1)
        sta = sta.view(N, sta.shape[1], self.scale, self.scale)
        return sta


class PTFEM(nn.Module):
    def __init__(self):
        super(PTFEM, self).__init__()
        self.conv = ConvBNReLU(512, 256, 1, 1, 0, has_bn=False, has_relu=False)
        self.qco_1 = QCO_2d(1, 8)
        self.qco_2 = QCO_2d(2, 8)
        self.qco_3 = QCO_2d(4, 8)
        self.qco_6 = QCO_2d(8, 8)
        # self.out = ConvBNReLU(512, 512, 1, 1, 0)
    def forward(self, x):
        H, W = x.shape[2:]
        x = self.conv(x)# 256,32,32
        sta_1 = self.qco_1(x) 
        sta_2 = self.qco_2(x)
        sta_3 = self.qco_3(x)
        sta_6 = self.qco_6(x)   
        N, C = sta_1.shape[:2]
        sta_1 = sta_1.view(N, C, 1, 1)
        sta_2 = sta_2.view(N, C, 2, 2)
        sta_3 = sta_3.view(N, C, 4, 4)
        sta_6 = sta_6.view(N, C, 8, 8)
        sta_1 = F.interpolate(sta_1, size=(H, W), mode='bilinear', align_corners=True)
        sta_2 = F.interpolate(sta_2, size=(H, W), mode='bilinear', align_corners=True)
        sta_3 = F.interpolate(sta_3, size=(H, W), mode='bilinear', align_corners=True)
        sta_6 = F.interpolate(sta_6, size=(H, W), mode='bilinear', align_corners=True)
        x = torch.cat([sta_1, sta_2, sta_3, sta_6], dim=1) 
        return x


if __name__=='__main__':
    x = torch.randn(2, 512, 32, 32).cuda()
    model = PTFEM().cuda()
    y = model(x)
    print(y.shape) # torch.Size([2, 512, 32, 32])