"""Model definition for Siamese UNet change detection"""

# Imports
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Two consecutive conv-BN-ReLU layers"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
    
class Encoder(nn.Module):
    """Contracting path for both images. Returns feature maps at each scale, plus bottleneck"""

    def __init__(self, in_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        c = base_channels

        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)

        self.bottleneck = ConvBlock(c * 8, c * 16)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple:
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        bn = self.bottleneck(self.pool(s4))

        return s1, s2, s3, s4, bn
    
class Decoder(nn.Module):
    """Expanding path, reconstructs change mask from difference features.
    Bilinear upsample -> concatenate skip connection -> conv block"""

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        c = base_channels

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = ConvBlock(c * 16 + c * 8, c * 8)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = ConvBlock(c * 8 + c * 4, c * 4)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = ConvBlock(c * 4 + c * 2, c * 2)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = ConvBlock(c * 2 + c, c)

    def forward(
        self,
        bottleneck: torch.Tensor,
        skip4: torch.Tensor,
        skip3: torch.Tensor,
        skip2: torch.Tensor,
        skip1: torch.Tensor
    ) -> torch.Tensor:
        
        x = torch.cat([self.up4(bottleneck), skip4], dim=1)
        x = self.dec4(x)
        x = torch.cat([self.up3(x), skip3], dim=1)
        x = self.dec3(x)
        x = torch.cat([self.up2(x), skip2], dim=1)
        x = self.dec2(x)
        x = torch.cat([self.up1(x), skip1], dim=1)
        x = self.dec1(x)

        return x
    
class SiameseUNet(nn.Module):
    """Siamese UNet architecture for change detection"""

    def __init__(self, in_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, base_channels)
        self.decoder = Decoder(base_channels)
        self.head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        # Encode both images
        s1_a, s2_a, s3_a, s4_a, bn_a = self.encoder(img_a)
        s1_b, s2_b, s3_b, s4_b, bn_b = self.encoder(img_b)

        # Compute absolute difference of bottleneck features
        diff_bn = torch.abs(bn_a - bn_b)
        diff_s4 = torch.abs(s4_a - s4_b)
        diff_s3 = torch.abs(s3_a - s3_b)
        diff_s2 = torch.abs(s2_a - s2_b)
        diff_s1 = torch.abs(s1_a - s1_b)

        # Decode with skip connections from image A
        x = self.decoder(diff_bn, diff_s4, diff_s3, diff_s2, diff_s1)

        return self.head(x)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)