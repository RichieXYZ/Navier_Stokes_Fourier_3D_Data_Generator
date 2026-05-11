import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvBlock3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, padding_mode="circular"),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, padding_mode="circular"),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class PixelShuffle3d(nn.Module):
    def __init__(self, upscale_factor=None):
        super().__init__()

        if upscale_factor is None:
            raise TypeError('__init__() missing 1 required positional argument: \'upscale_factor\'')

        self.upscale_factor = upscale_factor

    def forward(self, x):
        if x.ndim < 3:
            raise RuntimeError(
                f'pixel_shuffle expects input to have at least 3 dimensions, but got input with {x.ndim} dimension(s)'
            )
        elif x.shape[-4] % self.upscale_factor**3 != 0:
            raise RuntimeError(
                f'pixel_shuffle expects its input\'s \'channel\' dimension to be divisible by the cube of upscale_factor, but input.size(-4)={x.shape[-4]} is not divisible by {self.upscale_factor**3}'
            )

        channels, in_depth, in_height, in_width = x.shape[-4:]
        nOut = channels // self.upscale_factor ** 3

        out_depth = in_depth * self.upscale_factor
        out_height = in_height * self.upscale_factor
        out_width = in_width * self.upscale_factor

        input_view = x.contiguous().view(
            *x.shape[:-4],
            nOut,
            self.upscale_factor,
            self.upscale_factor,
            self.upscale_factor,
            in_depth,
            in_height,
            in_width
        )

        axes = torch.arange(input_view.ndim)[:-6].tolist() + [-3, -6, -2, -5, -1, -4]
        output = input_view.permute(axes).contiguous()

        return output.view(*x.shape[:-4], nOut, out_depth, out_height, out_width)

class UNetSR3d_V0(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        resolution_factor=8,
        encoder_depth=5,
        decoder_channels=None,
    ):
        super().__init__()

        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32, 16]
        assert encoder_depth == len(decoder_channels), \
            "encoder_depth must match length of decoder_channels"

        self.r = resolution_factor
        self.out_channels = out_channels
        self.depth = encoder_depth

        # Encoder
        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in decoder_channels[::-1]:
            self.encoders.append(ConvBlock3d(prev_ch, ch))
            prev_ch = ch

        self.pool = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = ConvBlock3d(
            decoder_channels[0],
            decoder_channels[0] * 2
        )

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for ch in decoder_channels:
            self.upconvs.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                    nn.Conv3d(ch * 2, ch, kernel_size=3, padding=1, padding_mode="circular")
                )
            )
            self.decoders.append(
                ConvBlock3d(ch * 2, ch)
            )

        # SR head
        self.final_conv = nn.Conv3d(
            decoder_channels[-1],
            out_channels * self.r ** 3,
            kernel_size=1
        )
        self.upsamples = nn.ModuleList()
        assert np.log2(self.r)%1==0, "resolution factor mus be a power of 2"
        for i in range(int(np.log2(self.r))):
            self.upsamples.append(
                PixelShuffle3d(2)
            )

    def forward(self, x):

        residual = x
        # Encoder
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skips = skips[::-1]
        for i in range(self.depth):
            x = self.upconvs[i](x)
            x = torch.cat([x, skips[i]], dim=1)
            x = self.decoders[i](x)

        # Super-resolution head
        x = self.final_conv(x)
        for up in self.upsamples:
            x = up(x)

        # Residual upsample
        residual = F.interpolate(
            residual,
            scale_factor=self.r,
            mode='trilinear',
            align_corners=False
        )

        return residual + x


class UNetSR3d(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        resolution_factor=8,
        encoder_depth=5,
        decoder_channels=None,
    ):
        super().__init__()

        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32, 16]
        assert encoder_depth == len(decoder_channels), \
            "encoder_depth must match length of decoder_channels"

        self.r = resolution_factor
        self.out_channels = out_channels
        self.depth = encoder_depth

        # Encoder
        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in decoder_channels[::-1]:
            self.encoders.append(ConvBlock3d(prev_ch, ch))
            prev_ch = ch

        self.pool = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = ConvBlock3d(
            decoder_channels[0],
            decoder_channels[0] * 2
        )

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for ch in decoder_channels:
            self.upconvs.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                    nn.Conv3d(ch * 2, ch, kernel_size=3, padding=1, padding_mode="circular")
                )
            )
            self.decoders.append(
                ConvBlock3d(ch * 2, ch)
            )

        # SR head
        self.sr_stages = nn.ModuleList()

        in_ch = decoder_channels[-1]
        assert np.log2(self.r) % 1 == 0, "resolution factor mus be a power of 2"
        for _ in range(int(np.log2(self.r))):
            self.sr_stages.append(
                nn.Sequential(
                    nn.Conv3d(in_ch, in_ch * 8, kernel_size=3, padding=1),
                    PixelShuffle3d(2),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, x):

        residual = x
        # Encoder
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skips = skips[::-1]
        for i in range(self.depth):
            x = self.upconvs[i](x)
            x = torch.cat([x, skips[i]], dim=1)
            x = self.decoders[i](x)

        # Super-resolution head
        for up in self.sr_stages:
            x = up(x)

        # Residual upsample
        residual = F.interpolate(
            residual,
            scale_factor=self.r,
            mode='trilinear',
            align_corners=False
        )

        return residual + x

def main():

    model = UNetSR3d(in_channels=1,
                   out_channels=1,
                   resolution_factor=16,
                   encoder_depth=3,
                   decoder_channels=[64, 32, 16],
                   )
    print(model)
    random_3d_in = torch.rand(32,32,32)
    print(random_3d_in.shape)

    y = model(random_3d_in.unsqueeze(0).unsqueeze(0))
    print(y.shape)

if __name__ == "__main__":
    main()
