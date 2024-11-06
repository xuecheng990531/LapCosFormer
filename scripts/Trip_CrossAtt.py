import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionModule(nn.Module):
    def __init__(self, in_channels_img=3, in_channels_aux=1, out_channels=64):
        super(CrossAttentionModule, self).__init__()
        # 调整通道数的 1x1 卷积
        self.img_conv = nn.Conv2d(in_channels_img, out_channels, kernel_size=1)
        # self.trimap_conv = nn.Conv2d(in_channels_aux, out_channels, kernel_size=1)
        self.edge_map_conv = nn.Conv2d(in_channels_aux, out_channels, kernel_size=1)
        
        # 用于生成查询、键和值的 1x1 卷积
        self.q_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.k_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.v_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        # self.gate_network = nn.Sequential(
        #     nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(out_channels, 2, kernel_size=1),  # 输出两个通道，分别对应两个输入的权重
        #     nn.Softmax(dim=1)  # 使用Softmax归一化权重
        # )
        self.gate_network = nn.Sequential(
            nn.Conv2d(out_channels*3, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, 3, kernel_size=1),  # 输出3个通道，分别对应两个输入的权重
            nn.Softmax(dim=1)  # 使用Softmax归一化权重
        )

        self.conv=nn.Sequential(nn.Conv2d(64,128,3,stride=2,padding=1),nn.BatchNorm2d(128),nn.ReLU())

    def forward(self, img, edge_map):
        # 通道对齐
        img = self.img_conv(img)  # (B, C, H, W)
        # trimap = self.trimap_conv(trimap)  # (B, C, H, W)
        edge_map = self.edge_map_conv(edge_map)  # (B, C, H, W)
        
        # 查询、键和值
        # Q_trimap = self.q_conv(trimap)  # (B, C, H, W)
        Q_edge = self.q_conv(edge_map)  # (B, C, H, W)
        K_img = self.k_conv(img)  # (B, C, H, W)
        V_img = self.v_conv(img)  # (B, C, H, W)
        
        # 计算 trimap Attention
        # S_trimap = torch.softmax((Q_trimap * K_img),dim=1)  # 使用逐元素乘法计算分数
        # Output_trimap = S_trimap * V_img 

        # 计算 edge map Attention
        S_edge = torch.softmax((Q_edge * K_img),dim=1)  # 使用逐元素乘法计算分数
        Output_edge = S_edge * V_img  

        # 门控网络计算权重
        # combined = torch.cat([Output_trimap, Output_edge], dim=1)  # 将两个输出在通道维度拼接
        combined = torch.cat([img,edge_map, Output_edge], dim=1)  # 将两个输出在通道维度拼接
        weights = self.gate_network(combined)  # (B, 2, H, W)，两个通道分别对应 trimap 和 edge 的权重
        
        # 分离权重
        # weight_trimap = weights[:, 0:1, :, :]  # (B, 1, H, W)
        weight_edge = weights[:, 1:2, :, :]  # (B, 1, H, W)
        
        # 计算加权输出
        # weighted_output_trimap = weight_trimap * Output_trimap
        weighted_output_edge = weight_edge * Output_edge
        
        # 最终输出
        # final_output = self.conv(weighted_output_trimap + weighted_output_edge)
        final_output = self.conv(weighted_output_edge)

        return final_output

# 示例使用
if __name__ == "__main__":
    # 128,128
    # 随机生成输入数据
    img = torch.randn(1, 3, 256, 256)  # 假设有两个batch, 3通道图像
    edge_map = torch.randn(1, 1, 256, 256)  # 单通道边缘图

    # 实例化并前向传播
    model = CrossAttentionModule()
    final = model(img, edge_map)

    print(final.shape)  # 检查输出的形状
