import torch
import torch.nn as nn
import torch.nn.functional as F

class ColorFeatureExtractor(nn.Module):
    def __init__(self):
        super(ColorFeatureExtractor, self).__init__()

    def forward(self, image):
        # 使用 RGB 通道的平均值作为颜色特征
        color_features = torch.mean(image, dim=(2, 3))
        return color_features

class SemanticDifferenceConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(SemanticDifferenceConvolution, self).__init__()
        self.color_feature_extractor = ColorFeatureExtractor()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.semantic_similarity = nn.ModuleList([nn.Conv2d(in_channels, 1, kernel_size, padding=kernel_size // 2) for _ in range(in_channels)])

    def forward(self, U, V):
        # 计算像素差异项
        pixel_difference = U.unsqueeze(1) - U.unsqueeze(2)
        pixel_difference = pixel_difference.sum(dim=2) 

        # 计算语义相似度项
        semantic_similarity = torch.cat([s(V) for s in self.semantic_similarity], dim=1)  # [batch_size, in_channels, height, width]

        # 对 pixel_difference 和 semantic_similarity 进行卷积操作
        combined_features = pixel_difference * semantic_similarity  # 将两者结合
        conv_result = self.conv(combined_features)  # 经过卷积，输出维度为 [batch_size, out_channels, height, width]

        # 计算颜色差异项
        color_U = self.color_feature_extractor(U)  # [batch_size, in_channels]
        color_V = self.color_feature_extractor(V)  # [batch_size, in_channels]
        color_difference = color_U.unsqueeze(2) - color_V.unsqueeze(1)  # [batch_size, in_channels, in_channels]

        # 我们只需要 color_difference 的每个通道之间的差异
        color_difference = torch.mean(color_difference, dim=1)  # 通过均值或其他方式减少维度 [batch_size, in_channels]

        # 将 color_difference 扩展为与 conv_result 相同的维度
        color_difference = color_difference.unsqueeze(-1).unsqueeze(-1)  # [batch_size, out_channels, 1, 1]
        color_difference = color_difference.expand(-1, conv_result.size(1), conv_result.size(2), conv_result.size(3))  # [batch_size, out_channels, height, width]

        # 最终将卷积结果与 color_difference 相乘
        output = conv_result * color_difference  # 结合颜色差异的影响
        return output

# # 特征融合
# class FeatureFusion(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(FeatureFusion, self).__init__()
#         self.conv = nn.Conv2d(in_channels*2, out_channels, kernel_size=1)

#     def forward(self, U, Y):
#         # 拼接输入特征图U和SDC输出Y
#         fused = torch.cat([U, Y], dim=1)
#         # 进行卷积操作实现特征融合
#         output = self.conv(fused)
#         return output

# SDN模块
class SDN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SDN, self).__init__()
        self.sdc = SemanticDifferenceConvolution(in_channels, out_channels)
        self.conv=nn.Conv2d(64, 16, kernel_size=1)
        # self.feature_fusion = FeatureFusion(in_channels, out_channels)

    def forward(self, xa, x):
        # 对U进行上采样
        xa_resized = F.interpolate(xa, size=(512, 512), mode='bilinear', align_corners=False)
        # 调整通道数，使其与x一致
        
        xa_resized = self.conv(xa_resized)# 1, 16, 512, 512
        Y = self.sdc(xa_resized, x)
        return Y

if __name__ == "__main__":
    x = torch.randn(1, 16, 512, 512).cuda()
    xa = torch.randn(1, 64, 256, 256).cuda()
    sdn = SDN(16, 16).cuda()
    output = sdn(xa, x)
    print(output.shape)