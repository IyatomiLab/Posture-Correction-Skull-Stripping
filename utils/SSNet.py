from typing import List

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, sample: str, batchnorm: bool, dropout: bool, activation: nn.modules.activation):
        super(ConvBlock, self).__init__()
        if sample == 'down':
            self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1, bias=False)
        elif sample == 'up':
            self.conv = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1, bias=False)
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)

        self.batchnorm = nn.BatchNorm2d(ch_out) if batchnorm else None
        self.dropout = nn.Dropout2d() if dropout else None
        self.activation = activation
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        h = self.conv(x)
        if self.batchnorm is not None:
            h = self.batchnorm(h)
        if self.dropout is not None:
            h = self.dropout(h)
        h = self.activation(h)
        return h

class Encoder(nn.Module):
    def __init__(self, ch_in: int):
        super(Encoder, self).__init__()
        self.conv0 = nn.Conv2d(ch_in, 64, kernel_size=3, stride=1, padding=1, bias=True)
        nn.init.normal_(self.conv0.weight, mean=0.0, std=0.02)

        self.conv1 = self.make_downblock(64, 128)
        self.conv2 = self.make_downblock(128, 256)
        self.conv3 = self.make_downblock(256, 512)
        self.conv4 = self.make_downblock(512, 512)
        self.conv5 = self.make_downblock(512, 512)
        self.conv6 = self.make_downblock(512, 512)
        self.conv7 = self.make_downblock(512, 512)

    def make_downblock(self, ch_in: int, ch_out: int) -> ConvBlock:
        return ConvBlock(ch_in=ch_in, ch_out=ch_out, sample='down', batchnorm=True, dropout=False, activation=nn.LeakyReLU())
    
    def forward(self, x: torch.tensor) -> list:
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)
        return [conv0,conv1,conv2,conv3,conv4,conv5,conv6,conv7]

class Decoder(nn.Module):
    def __init__(self, ch_out: int) -> None:
        super(Decoder, self).__init__()
        self.conv7 = self.make_upblock(512, 512, dropout=True)
        self.conv6 = self.make_upblock(1024, 512, dropout=True)
        self.conv5 = self.make_upblock(1024, 512, dropout=True)
        self.conv4 = self.make_upblock(1024, 512, dropout=False)
        self.conv3 = self.make_upblock(1024, 256, dropout=False)
        self.conv2 = self.make_upblock(512, 128, dropout=False)
        self.conv1 = self.make_upblock(256, 64, dropout=False)
        self.conv0 = nn.Conv2d(128, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        
        nn.init.normal_(self.conv0.weight, mean=0.0, std=0.02)

    def make_upblock(self, in_channels: int, out_channels: int, dropout=False) -> ConvBlock:
        return ConvBlock(in_channels, out_channels, sample='up', batchnorm=True, dropout=dropout, activation=nn.ReLU())

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        conv7_result = self.conv7(xs[7])
        conv6_result = self.conv6(torch.cat([conv7_result, xs[6]], dim=1))
        conv5_result = self.conv5(torch.cat([conv6_result, xs[5]], dim=1))
        conv4_result = self.conv4(torch.cat([conv5_result, xs[4]], dim=1))
        conv3_result = self.conv3(torch.cat([conv4_result, xs[3]], dim=1))
        conv2_result = self.conv2(torch.cat([conv3_result, xs[2]], dim=1))
        conv1_result = self.conv1(torch.cat([conv2_result, xs[1]], dim=1))
        conv0_result = self.conv0(torch.cat([conv1_result, xs[0]], dim=1))
        return conv0_result

class SSNet(nn.Module):
    def __init__(self, ch_in: int) -> None:
        super(SSNet, self).__init__()
        self.encoder = Encoder(ch_in=ch_in)
        self.decoder = Decoder(ch_out=ch_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(x)
        prob_map = self.decoder(embedding)
        return prob_map