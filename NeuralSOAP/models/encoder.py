import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(
        self,
        bottleneck_channel: int,
        in_channels: int,
        level_channels: list,
        smaller: bool = False,
        n_species: int = 5,  # 可选：元素数，默认 5 (QM9)
        n_max: int = 8,      # 可选：SOAP n_max
        l_max: int = 6,      # 可选：SOAP l_max
        soap_reduced_dim: int = 512  # 新增：SOAP 降维目标维度（端到端学习）
    ):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.level_channels = level_channels
        self.bottleneck_channel = bottleneck_channel
        self.smaller = smaller

        # Dynamically create conv layers
        self.convs = nn.ModuleList()
        current_ch = in_channels
        for ch in level_channels:
            self.convs.append(nn.Conv3d(current_ch, ch, kernel_size=3, padding=1))
            current_ch = ch

        # Pool after all but the last conv (to avoid size=0)
        self.pools = nn.ModuleList([nn.MaxPool3d(2) for _ in range(len(level_channels) - 1)])

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        cnn_dim = level_channels[-1]
        self.fc_cnn = nn.Linear(cnn_dim, bottleneck_channel)

        # 计算原始 SOAP 维度
        self.soap_dim = self._calculate_soap_dim(n_species, n_max, l_max)  # 5740

        # 新增：端到端可训练降维层（线性 + ReLU + LayerNorm，避免爆炸）
        self.soap_reducer = nn.Sequential(
            nn.Linear(self.soap_dim, soap_reduced_dim),
            nn.ReLU(),
            nn.LayerNorm(soap_reduced_dim)  # 规范化稳定梯度
        )

        # 融合层使用降维后维度
        self.fc_fusion = nn.Linear(cnn_dim + soap_reduced_dim, bottleneck_channel)

    def _calculate_soap_dim(self, n_species, n_max, l_max):
        """动态计算 dscribe.SOAP 特征维度"""
        n_pairs_diag = n_species  # 对角对 (Z==Z')
        n_pairs_off = n_species * (n_species - 1) // 2  # 非对角对 (Z<Z')
        n_terms_diag = n_max * (n_max + 1) // 2  # 上三角 (包括对角线)
        n_terms_off = n_max * n_max  # 全矩阵
        n_l = l_max + 1
        total = (n_pairs_diag * n_terms_diag + n_pairs_off * n_terms_off) * n_l
        return total

    def forward(self, x: torch.Tensor, soap_features: torch.Tensor = None) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x))
            if i < len(self.pools):  # Pool only for first len-1 convs
                x = self.pools[i](x)

        x = self.global_pool(x).squeeze(-1).squeeze(-1).squeeze(-1)

        # 添加调试打印
        #print(f"[DEBUG] CNN features shape after global_pool: {x.shape}")
        if soap_features is not None:
            # 端到端降维 SOAP
            soap_reduced = self.soap_reducer(soap_features)
            #print(f"[DEBUG] SOAP reduced shape: {soap_reduced.shape}")
            x = torch.cat([x, soap_reduced], dim=1)
            #print(f"[DEBUG] Concatenated features shape: {x.shape}")
            x = self.fc_fusion(x)
        else:
            x = self.fc_cnn(x)

        return x