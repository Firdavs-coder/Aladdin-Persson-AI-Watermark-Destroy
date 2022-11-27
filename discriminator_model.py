import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class CNNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(
                in_channels, out_channels, 3, stride, 1, bias=False, padding_mode="reflect"
            )),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512, 512, 1024]):
        super().__init__()
        self.inital = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    in_channels*2,
                    features[0],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="reflect"
                )
            ),
            nn.LeakyReLU(0.2)
        )
        layers = []
        in_channels = features[0]
        for idx, feature in enumerate(features[1:]):
            layers.append(CNNBlock(in_channels, feature,
                          stride=1 if idx == len(features) else 2))
            in_channels = feature
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, kernel_size=3, stride=1,
                          padding=1, padding_mode="reflect")
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.inital(x)
        return self.model(x)
