import torch
import numpy as np
from torch import nn

from torchvision import models
from model.conformer_model import Conformer
import pdb
import torch.nn.functional as F

def init_layer(layer: nn.Module):
    r"""Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn: nn.Module):
    r"""Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)
    bn.running_mean.data.fill_(0.0)
    bn.running_var.data.fill_(1.0)


def act(x: torch.Tensor, activation: str) -> torch.Tensor:

    if activation == "relu":
        return F.relu_(x)

    elif activation == "leaky_relu":
        return F.leaky_relu_(x, negative_slope=0.01)

    elif activation == "swish":
        return x * torch.sigmoid(x)

    else:
        raise Exception("Incorrect activation!")

class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation, momentum):
        r"""Residual block."""
        super(ConvBlockRes, self).__init__()

        self.activation = activation
        padding = [kernel_size[0] // 2, kernel_size[1] // 2]

        self.bn1 = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            padding=padding,
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            padding=padding,
            bias=False,
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            )

            self.is_shortcut = True
        else:
            self.is_shortcut = False

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_layer(self.conv1)
        init_layer(self.conv2)

        if self.is_shortcut:
            init_layer(self.shortcut)

    def forward(self, x):
        origin = x
        x = self.conv1(act(self.bn1(x), self.activation))
        x = self.conv2(act(self.bn2(x), self.activation))

        if self.is_shortcut:
            return self.shortcut(origin) + x
        else:
            return origin + x


class EncoderBlockRes4B(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, downsample, activation, momentum
    ):
        r"""Encoder block, contains 8 convolutional layers."""
        super(EncoderBlockRes4B, self).__init__()

        self.conv_block1 = ConvBlockRes(
            in_channels, out_channels, kernel_size, activation, momentum
        )
        self.conv_block2 = ConvBlockRes(
            out_channels, out_channels, kernel_size, activation, momentum
        )
        # self.conv_block3 = ConvBlockRes(
        #     out_channels, out_channels, kernel_size, activation, momentum
        # )
        # self.conv_block4 = ConvBlockRes(
        #     out_channels, out_channels, kernel_size, activation, momentum
        # )
        self.downsample = downsample

    def forward(self, x):
        encoder = self.conv_block1(x)
        encoder = self.conv_block2(encoder)
        # encoder = self.conv_block3(encoder)
        # encoder = self.conv_block4(encoder)
        encoder_pool = F.avg_pool2d(encoder, kernel_size=self.downsample)
        return encoder_pool, encoder



class ResnetConformer(nn.Module):
    def __init__(self):
        super(ResnetConformer, self).__init__()
        
        activation = "relu"
        momentum = 0.01
        
        self.encoder_block1 = EncoderBlockRes4B(
            in_channels=8,
            out_channels=32,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block2 = EncoderBlockRes4B(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block3 = EncoderBlockRes4B(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )
        self.encoder_block4 = EncoderBlockRes4B(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            downsample=(2, 2),
            activation=activation,
            momentum=momentum,
        )


        self.conformer = Conformer(input_dim=256,
                      model_dim=256,
                      num_heads=8,
                      ffn_dim=1024,
                      num_layers=4,
                      depthwise_conv_kernel_size=31,
                      inptarget='MagMag')

        self.classifier = nn.Linear(256, 360)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        # pass the input through the resnet model
        x, _ = self.encoder_block1(x)
        x, _ = self.encoder_block2(x)
        x, _ = self.encoder_block3(x)
        x, _ = self.encoder_block4(x)

        x = x.permute(0, 2, 1, 3)

        # average pooling of the last dimension
        x = F.avg_pool2d(x, kernel_size=(1, x.size(3))).squeeze(3) # b, 16, 256

        length = torch.tensor([x.size(1)]).repeat(x.size(0)).to(x.device)
        # pdb.set_trace()
        # 
        # pass the output of the resnet model through the conformer model
        out, length = self.conformer(x, length)
        out = self.classifier(out)
        # pdb.set_trace()
        out = out.permute(0, 2, 1)
        # average pooling of the time dimension
        out = F.avg_pool1d(out, kernel_size=out.size(2)).squeeze(2)
        out = self.softmax(out)
        return out