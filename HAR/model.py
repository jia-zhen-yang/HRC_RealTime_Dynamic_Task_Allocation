import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, kernel_size=5, stride=1, dropout=0.5, residual=True):
        super(GraphConvBlock, self).__init__()
        self.A = A  # [V, V]
        self.V = A.shape[0]
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 可學習的鄰接矩陣係數
        self.A_weight = nn.Parameter(torch.ones_like(torch.tensor(A, dtype=torch.float32)))

        # 空間圖卷積：1x1 conv 對應不同鄰居聚合
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # 時間卷積：kernel_size x 1（只在時間軸上做）
        padding = (kernel_size - 1) // 2
        self.tcn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1),
                      padding=(padding, 0), stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 殘差連線
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride,1)),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [B, C, T, V]
        A = self.A.to(x.device) * self.A_weight  # 可學習權重
        x_gcn = self.gcn(torch.einsum('vu,nctu->nctv', A, x))
        x = self.tcn(x_gcn) + self.residual(x)
        return self.relu(x)


class STGCN(nn.Module):
    def __init__(self, in_channels, num_class, A, edge_importance_weighting=True):
        super(STGCN, self).__init__()
        self.data_bn = nn.BatchNorm1d(in_channels * A.shape[0])

        self.layers = nn.ModuleList([
            GraphConvBlock(in_channels, 64, A, residual=False),
            GraphConvBlock(64, 64, A),
            GraphConvBlock(64, 64, A),
            GraphConvBlock(64, 128, A, stride=2),
            GraphConvBlock(128, 128, A),
            GraphConvBlock(128, 256, A, stride=2),
            GraphConvBlock(256, 256, A)
        ])

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 最終輸出 [B, C, 1, 1]
        self.fc = nn.Linear(256, num_class)

    def forward(self, x):
        # x: [B, C, T, V]
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()  # [N, C, T, V]

        for layer in self.layers:
            x = layer(x)

        x = self.pool(x)  # [N, 256, 1, 1]
        x = x.view(x.size(0), -1)  # [N, 256]
        return self.fc(x)  # [N, num_class]
