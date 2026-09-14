import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN3D(nn.Module):
    """3D CNN编码器基础实现"""

    def __init__(self, in_channels=5, code_dim=256, grid_dim=32):
        super().__init__()
        self.code_dim = code_dim

        # 3D卷积层
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)

        # 池化层
        self.pool = nn.MaxPool3d(2)

        # 计算展平后的维度
        final_dim = grid_dim // (2 ** 4)  # 4次池化
        self.fc_input_dim = 256 * (final_dim ** 3)

        # 全连接层
        self.fc = nn.Linear(self.fc_input_dim, code_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x shape: [batch_size, channels, depth, height, width]
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = F.relu(self.conv4(x))
        x = self.pool(x)

        # 展平
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def get_output_dim(self):
        return self.code_dim


class FusionEncoder(nn.Module):
    """融合3D CNN和SOAP特征的编码器"""

    def __init__(self, cnn_config, soap_dim, code_dim, fusion_type='gated'):
        super().__init__()

        # 创建3D CNN编码器
        self.cnn_encoder = CNN3D(
            in_channels=cnn_config.get('in_channels', 5),
            code_dim=cnn_config.get('code_dim', 256),
            grid_dim=cnn_config.get('grid_dim', 32)
        )

        self.soap_dim = soap_dim
        self.fusion_type = fusion_type

        # 获取CNN编码器的输出维度
        cnn_output_dim = self.cnn_encoder.get_output_dim()

        if fusion_type == 'concat':
            # 简单拼接融合
            self.fusion_layer = nn.Sequential(
                nn.Linear(cnn_output_dim + soap_dim, code_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(code_dim, code_dim)
            )

        elif fusion_type == 'attention':
            # 注意力融合
            self.cnn_proj = nn.Linear(cnn_output_dim, code_dim)
            self.soap_proj = nn.Linear(soap_dim, code_dim)
            self.attention = nn.MultiheadAttention(code_dim, num_heads=8, batch_first=True)
            self.fusion_layer = nn.Sequential(
                nn.Linear(code_dim, code_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )

        elif fusion_type == 'gated':
            # 门控融合
            self.cnn_proj = nn.Linear(cnn_output_dim, code_dim)
            self.soap_proj = nn.Linear(soap_dim, code_dim)

            # 门控机制
            self.gate = nn.Sequential(
                nn.Linear(cnn_output_dim + soap_dim, code_dim),
                nn.Sigmoid()
            )

            self.fusion_layer = nn.Sequential(
                nn.Linear(code_dim, code_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")

    def forward(self, voxel_input, soap_features=None):
        """
        前向传播

        Args:
            voxel_input: 3D体素输入 [batch_size, channels, D, H, W]
            soap_features: SOAP特征 [batch_size, soap_dim]

        Returns:
            融合后的特征编码 [batch_size, code_dim]
        """
        # 3D CNN编码
        cnn_features = self.cnn_encoder(voxel_input)

        # 如果没有SOAP特征，直接返回CNN特征
        if soap_features is None:
            return cnn_features

        batch_size = cnn_features.size(0)

        if self.fusion_type == 'concat':
            # 简单拼接融合
            fused_features = torch.cat([cnn_features, soap_features], dim=1)
            output = self.fusion_layer(fused_features)

        elif self.fusion_type == 'attention':
            # 注意力融合
            cnn_proj = self.cnn_proj(cnn_features).unsqueeze(1)  # [B, 1, D]
            soap_proj = self.soap_proj(soap_features).unsqueeze(1)  # [B, 1, D]

            # 多头注意力
            features = torch.cat([cnn_proj, soap_proj], dim=1)  # [B, 2, D]
            attended, _ = self.attention(features, features, features)

            # 平均池化并通过最终层
            pooled = attended.mean(dim=1)  # [B, D]
            output = self.fusion_layer(pooled)

        elif self.fusion_type == 'gated':
            # 门控融合
            cnn_proj = self.cnn_proj(cnn_features)
            soap_proj = self.soap_proj(soap_features)

            # 计算门控权重
            gate_input = torch.cat([cnn_features, soap_features], dim=1)
            gate_weights = self.gate(gate_input)

            # 门控融合：学习如何平衡两种特征
            fused = gate_weights * cnn_proj + (1 - gate_weights) * soap_proj
            output = self.fusion_layer(fused)

        return output

    def get_output_dim(self):
        """返回输出维度"""
        return self.cnn_encoder.get_output_dim()


def create_fusion_encoder(config):
    """创建融合编码器的工厂函数"""
    return FusionEncoder(
        cnn_config=config.get('cnn', {}),
        soap_dim=config.get('soap_dim', 1680),  # SOAP特征的默认维度
        code_dim=config.get('code_dim', 256),
        fusion_type=config.get('fusion_type', 'gated')
    )