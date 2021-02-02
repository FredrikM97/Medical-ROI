import torch
import torch.nn as nn
import torch.nn.functional as F

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(2,  ceil_mode=True),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        print("down",x.shape)
        x = self.mpconv(x)
        print("down after",x.shape)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, trilinear=True):
        super(up, self).__init__()
        
        self.convtrans = nn.ConvTranspose3d(in_ch//2, in_ch//2, 2, stride=2)
        self.trilinear = trilinear

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, dec, enc):
        print(self.trilinear)
        #if self.trilinear:
        #    dec = nn.functional.interpolate(dec, scale_factor=2, mode='trilinear', align_corners=True)
        #else:
        dec = self.convtrans(dec)
        
        # input is CHW
        
        #diffZ = x2.size()[2] - x1.size()[2]
        #diffY = x2.size()[3] - x1.size()[3]
        #diffX = x2.size()[4] - x1.size()[4]
        
        #x1 = F.pad(x1, [diffZ // 2, diffZ - diffZ // 2,
        #                diffX // 2, diffX - diffX // 2,
        #                diffY // 2, diffY - diffY // 2])
        
        print("up",dec.shape, enc.shape)
        x = torch.cat([enc, dec], dim=1)
        x = self.conv(x)
        print("Up after",x.shape)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x



class UNet3D(nn.Module):
    """Standard U-Net architecture network.
    
    Input params:
        n_channels: Number of input channels (usually 1 for a grayscale image).
        n_classes: Number of output channels (2 for binary segmentation).
    """
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        print(x.shape)
        enc1 = self.inc(x)
        enc2 = self.down1(enc1)
        enc3 = self.down2(enc2)
        enc4 = self.down3(enc3)
        enc5 = self.down4(enc4)
        dec4 = self.up1(enc5, enc4)
        dec3 = self.up2(dec4, enc3)
        dec2 = self.up3(dec3, enc2)
        dec1 = self.up4(dec2, enc1)
        dec0 = self.outc(dec1)
        return x
