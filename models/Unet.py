import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):

    """Double convolutional block which consists of two CNA blocks"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        # if not mid_channels:
        #     mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Down block [MaxPool -> CNA -> CNA]"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.double_conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        output = self.max_pool(x)
        output = self.double_conv(output)
        return output


class Up(nn.Module):
    """Up block [Upsample -> CNA -> CNA]"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.double_conv = DoubleConv(
            in_channels = in_channels,
            out_channels = out_channels#,
            #mid_channels = in_channels//2
        )

    def forward(self, x, x_shortcut):
        output = self.upsample(x)
        diff_y = x_shortcut.size()[2] - output.size()[2]
        diff_x = x_shortcut.size()[3] - output.size()[3]

        output = F.pad(output, (diff_x//2, diff_x - diff_x//2, diff_y//2, diff_y - diff_y//2))

        output = torch.cat((x_shortcut, output), dim=1)
        output = self.double_conv(output)
        return output


class Unet(nn.Module):
    """
    Main Unet class

    Parameters
    ----------

    num_classes: int
        number of classes
    n_channels: int
        number of channels in original image
    """

    def __init__(self, num_classes, n_channels=3):
        super().__init__()

        self.first_conv = DoubleConv(in_channels=n_channels, out_channels=64)

        self.down1 = Down(in_channels=64, out_channels=128)
        self.down2 = Down(in_channels=128, out_channels=256)
        self.down3 = Down(in_channels=256, out_channels=512)
        self.down4 = Down(in_channels=512, out_channels=512)

        self.up1 = Up(in_channels=1024, out_channels=256)
        self.up2 = Up(in_channels=512, out_channels=128)
        self.up3 = Up(in_channels=256, out_channels=64)
        self.up4 = Up(in_channels=128, out_channels=64)

        self.last_conv = nn.Conv2d(
            in_channels=64,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        down1 = self.first_conv(x)

        down2 = self.down1(down1)
        down3 = self.down2(down2)
        down4 = self.down3(down3)
        down5 = self.down4(down4)

        up1 = self.up1(down5, down4)
        up2 = self.up2(up1, down3)
        up3 = self.up3(up2, down2)
        up4 = self.up4(up3, down1)

        output = self.last_conv(up4)
        return output
