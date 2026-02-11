import torch
import torch.nn as nn
import torch.nn.functional as F
from . import MobileNetV2

class SupervisedAttentionModule(nn.Module):
    def __init__(self, mid_d):
        super(SupervisedAttentionModule, self).__init__()
        self.mid_d = mid_d
        # fusion
        self.cls = nn.Conv2d(self.mid_d, 1, kernel_size=1)
        self.conv_context = nn.Sequential(
            nn.Conv2d(2, self.mid_d, kernel_size=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        mask = self.cls(x)
        mask_f = torch.sigmoid(mask)
        mask_b = 1 - mask_f
        context = torch.cat([mask_f, mask_b], dim=1)
        context = self.conv_context(context)
        x = x.mul(context)
        x_out = self.conv2(x)
        return x_out, mask


# Temporal-Dynamic Enhanced Interaction (TDEI) Module
class TDEI(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TDEI, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Temporal Cross-Interaction components
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
        # Dynamic Change Region Localization components
        self.mask_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.refine_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Activation functions
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        # Temporal Cross-Interaction
        diff = torch.abs(x1 - x2)  # Explicit temporal difference
        cross1 = self.relu(self.conv1(x1 + diff))  # Forward interaction
        cross2 = self.relu(self.conv2(x2 + diff))  # Backward interaction
        temporal_out = cross1 + cross2  # Combine results
        
        # Dynamic Change Region Localization
        mask = self.sigmoid(self.mask_conv(temporal_out))
        localized = temporal_out * mask
        refined = self.relu(self.refine_conv(localized))
        
        return refined, mask


# Decoder
class Decoder(nn.Module):
    def __init__(self, mid_d=64):
        super(Decoder, self).__init__()
        self.mid_d = mid_d
        # fusion
        self.sam_p5 = SupervisedAttentionModule(self.mid_d)
        self.sam_p4 = SupervisedAttentionModule(self.mid_d)
        self.sam_p3 = SupervisedAttentionModule(self.mid_d)
        self.conv_p4 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_p3 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.cls = nn.Conv2d(self.mid_d, 1, kernel_size=1)

    def forward(self, d2, d3, d4, d5):
        # high-level
        p5, mask_p5 = self.sam_p5(d5)
        p4 = self.conv_p4(d4 + F.interpolate(p5, scale_factor=(2, 2), mode='bilinear'))

        p4, mask_p4 = self.sam_p4(p4)
        p3 = self.conv_p3(d3 + F.interpolate(p4, scale_factor=(2, 2), mode='bilinear'))

        p3, mask_p3 = self.sam_p3(p3)
        p2 = self.conv_p2(d2 + F.interpolate(p3, scale_factor=(2, 2), mode='bilinear'))
        mask_p2 = self.cls(p2)

        return p2, p3, p4, p5, mask_p2, mask_p3, mask_p4, mask_p5


# 主模型：TILNet
class TILNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=1):
        super(TILNet, self).__init__()
        self.backbone = MobileNetV2.mobilenet_v2(pretrained=True)

        self.backbone_channels = [24, 32, 96, 1280]  # 删除了第一个16通道
        self.mid_d = 64  # 统一的中间维度

        # 通道适配层：将不同阶段的特征统一到 mid_d
        self.adapt1 = nn.Conv2d(self.backbone_channels[0], self.mid_d, kernel_size=1)  # 24 -> 64
        self.adapt2 = nn.Conv2d(self.backbone_channels[1], self.mid_d, kernel_size=1)  # 32 -> 64
        self.adapt3 = nn.Conv2d(self.backbone_channels[2], self.mid_d, kernel_size=1)  # 96 -> 64
        self.adapt4 = nn.Conv2d(self.backbone_channels[3], self.mid_d, kernel_size=1)  # 1280 -> 64

        # TDEI - Temporal-Dynamic Enhanced Interaction modules
        self.tdei1 = TDEI(self.mid_d, self.mid_d)
        self.tdei2 = TDEI(self.mid_d, self.mid_d)
        self.tdei3 = TDEI(self.mid_d, self.mid_d)
        self.tdei4 = TDEI(self.mid_d, self.mid_d)

        # Decoder
        self.decoder = Decoder(self.mid_d)

    def forward(self, x1, x2):
        # Backbone 特征提取
        x1_feats = self.extract_features(x1)
        x2_feats = self.extract_features(x2)

        # 通道适配
        x1_feats[0] = self.adapt1(x1_feats[0])  # 24 channels
        x1_feats[1] = self.adapt2(x1_feats[1])  # 32 channels
        x1_feats[2] = self.adapt3(x1_feats[2])  # 96 channels
        x1_feats[3] = self.adapt4(x1_feats[3])  # 1280 channels
        
        x2_feats[0] = self.adapt1(x2_feats[0])
        x2_feats[1] = self.adapt2(x2_feats[1])
        x2_feats[2] = self.adapt3(x2_feats[2])
        x2_feats[3] = self.adapt4(x2_feats[3])

        # TDEI: Temporal-Dynamic Enhanced Interaction
        c1_refined, _ = self.tdei1(x1_feats[0], x2_feats[0])
        c2_refined, _ = self.tdei2(x1_feats[1], x2_feats[1])
        c3_refined, _ = self.tdei3(x1_feats[2], x2_feats[2])
        c4_refined, _ = self.tdei4(x1_feats[3], x2_feats[3])

        # Decoder
        _, _, _, _, mask_p2, mask_p3, mask_p4, mask_p5 = self.decoder(
            c1_refined, c2_refined, c3_refined, c4_refined
        )

        # Final change map (combined from all stages)
        _, _, H, W = x1.shape
        
        mask_p2 = F.interpolate(mask_p2, size=(H, W), mode='bilinear', align_corners=False)
        mask_p2 = torch.sigmoid(mask_p2)
        mask_p3 = F.interpolate(mask_p3, size=(H, W), mode='bilinear', align_corners=False)
        mask_p3 = torch.sigmoid(mask_p3)
        mask_p4 = F.interpolate(mask_p4, size=(H, W), mode='bilinear', align_corners=False)
        mask_p4 = torch.sigmoid(mask_p4)
        mask_p5 = F.interpolate(mask_p5, size=(H, W), mode='bilinear', align_corners=False)
        mask_p5 = torch.sigmoid(mask_p5)

        return mask_p2, mask_p3, mask_p4, mask_p5

    def extract_features(self, x):
        features = []
        
        for i, layer in enumerate(self.backbone.features):
            x = layer(x)
            if i == 3:  # 24 channels, stride=4
                features.append(x)
            elif i == 6:  # 32 channels, stride=8
                features.append(x)
            elif i == 13:  # 96 channels, stride=16
                features.append(x)
            elif i == 18:  # 1280 channels, stride=32
                features.append(x)
        
        return features


if __name__ == "__main__":
    # 做个小测试，确保维度正确
    model = TILNet()
    x1 = torch.randn(2, 3, 256, 256)
    x2 = torch.randn(2, 3, 256, 256)
    
    outputs = model(x1, x2)
    print(f"Number of outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"Output {i+1} shape: {out.shape}")