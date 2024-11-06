import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import cv2

def unfold2d(x, kernel_size, stride, padding):
    x = F.pad(x, [padding]*4)
    bs, in_c, h, w = x.size()
    ks = kernel_size
    strided_x = x.as_strided((bs, in_c, (h - ks) // stride + 1, (w - ks) // stride + 1, ks, ks),
        (in_c * h * w, h * w, stride * w, stride, w, 1))
    return strided_x

class CosSim2d(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=1, stride=1,
        padding=0, eps=1e-12, bias=True):
        super(CosSim2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.eps = eps
        self.padding = padding

        w = torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        nn.init.xavier_normal_(w)
        self.w = nn.Parameter(w.view(out_channels, in_channels, -1), requires_grad=True)
        
        self.p = nn.Parameter(torch.empty(out_channels))
        nn.init.constant_(self.p, 2)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            nn.init.constant_(self.bias, 0)
        else:
            self.bias = None
    
    def sigplus(self, x):
        return nn.Sigmoid()(x) * nn.Softplus()(x)
        
    def forward(self, x):
        x = unfold2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding) # nchwkk
        n, c, h, w, _, _ = x.shape
        x = x.reshape(n,c,h,w,-1)
        x = F.normalize(x, p=2.0, dim=-1, eps=self.eps)

        w = F.normalize(self.w, p=2.0, dim=-1, eps=self.eps)
        x = torch.einsum('nchwl,vcl->nvhw', x, w)
        sign = torch.sign(x)

        x = torch.abs(x) + self.eps
        x = x.pow(self.sigplus(self.p).view(1, -1, 1, 1))
        x = sign * x

        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1)
        return x


class LaplacianPyramid(nn.Module):
    def __init__(self, in_channels=64, pyramid_levels=3):
        """
        Constructs a Laplacian pyramid from an input tensor.

        Args:
            in_channels    (int): Number of input channels.
            pyramid_levels (int): Number of pyramid levels.
        
        Input: 
            x : (B, in_channels, H, W)
        Output:
            Fused frequency attention map : (B, in_channels, in_channels)
        """
        super().__init__()
        self.in_channels = in_channels
        self.pyramid_levels = pyramid_levels
        sigma = 1.6
        s_value = 2 ** (1/3)

        self.sigma_kernels = [
            self.get_gaussian_kernel(2*i + 3, sigma * s_value ** i)
            for i in range(pyramid_levels)
        ]

    def get_gaussian_kernel(self, kernel_size, sigma):
        kernel_weights = cv2.getGaussianKernel(ksize=kernel_size, sigma=sigma)
        kernel_weights = kernel_weights * kernel_weights.T
        kernel_weights = np.repeat(kernel_weights[None, ...], self.in_channels, axis=0)[:, None, ...]

        return torch.from_numpy(kernel_weights).float().cuda()

    def forward(self, x):
        G = x
        
        # # Level 1
        # L0 = Rearrange('b d h w -> b d (h w)')(G)
        # L0_att= F.softmax(L0, dim=2) @ L0.transpose(1, 2)  # L_k * L_v
        # L0_att = F.softmax(L0_att, dim=-1)
        
        # Next Levels
        # attention_maps = [L0_att]
        attention_maps = []
        pyramid = [G]
        for kernel in self.sigma_kernels:
            G = F.conv2d(input=G, weight=kernel, bias=None, padding='same', groups=self.in_channels)
            pyramid.append(G)
        
        for i in range(1, self.pyramid_levels):
            L = torch.sub(pyramid[i - 1], pyramid[i])
            # L = Rearrange('b d h w -> b d (h w)')(L)
            # L_att= F.softmax(L, dim=2) @ L.transpose(1, 2) 
            # attention_maps.append(L_att)
            attention_maps.append(L)
        return sum(attention_maps)





class LapCos_Block(nn.Module):
    """
    args:
        in_channels:    (int) : Embedding Dimension.
        key_channels:   (int) : Key Embedding Dimension,   Best: (in_channels).
        value_channels: (int) : Value Embedding Dimension, Best: (in_channels or in_channels//2). 
        pyramid_levels  (int) : Number of pyramid levels.
    input:
        x : [B, D, H, W]
    output:
        Efficient Attention : [B, D, H, W]
    
    """
    
    def __init__(self, in_channels, key_channels, value_channels, pyramid_levels=3):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1) 
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)
        
        # Build a laplacian pyramid
        self.freq_attention = LaplacianPyramid(in_channels=in_channels, pyramid_levels=pyramid_levels) 
        self.cos_sim = CosSim2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        n, _, h, w = x.size()

        x_cosine_sim=self.cos_sim(x)
        
        # 关于x自身的self attention
        keys = F.softmax(self.keys(x_cosine_sim).reshape((n, self.key_channels, h * w)), dim=2)
        queries = F.softmax(self.queries(x_cosine_sim).reshape(n, self.key_channels, h * w), dim=1)
        values = self.values(x_cosine_sim).reshape((n, self.value_channels, h * w))      
        context = keys @ values.transpose(1, 2) # dk*dv            
        attended_value = (context.transpose(1, 2) @ queries).reshape(n, self.value_channels, h, w) # n*dv
        eff_attention  = self.reprojection(attended_value)

        # Freqency Attention
        freq_context = self.freq_attention(x)

        attention = torch.add(freq_context, eff_attention)
        results=torch.matmul(attention,x)+x
    
        return results

    

if __name__=='__main__':
    x=torch.randn(2, 1024, 16, 16).cuda()
    model=LapCos_Block(in_channels=1024, key_channels=1024, value_channels=1024, pyramid_levels=3).cuda()
    print(model(x).shape)