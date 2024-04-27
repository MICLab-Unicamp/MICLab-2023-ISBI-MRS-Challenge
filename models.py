"""
Maintainer: Gabriel Dias (g172441@dac.unicamp.br)
            Mateus Oliveira (m203656@dac.unicamp.br)
"""

import torch.nn as nn
import timm
import torch


def get_n_out_features(encoder, img_size, nchannels):
    out_feature = encoder(torch.randn(1, nchannels, img_size[0], img_size[1]))
    n_out = 1
    for dim in out_feature[-1].shape:
        n_out *= dim
    return n_out


class SpectroViT(nn.Module):
    def __init__(self, timm_network: str = "vit_base_patch32_224",
                 image_size: tuple[int, int] = (224, 224),
                 nchannels: int = 3,
                 pretrained: bool = True,
                 num_classes: int = 0,
                 ):
        super().__init__()

        model_creator = {'model_name': timm_network,
                         "pretrained": pretrained,
                         "num_classes": num_classes}

        self.encoder = timm.create_model(**model_creator)

        self.dimensionality_reductor = None

        for param in self.encoder.parameters():
            param.requires_grad = True

        n_out = get_n_out_features(self.encoder, image_size, nchannels)

        self.dimensionality_up_sampling = nn.Sequential(
            nn.Linear(n_out, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 2048)
        )

    def forward(self, signal_input):
        output = self.encoder(signal_input)
        output = self.dimensionality_up_sampling(output)

        return output


class SpectroViTTrack3(nn.Module):
    def __init__(self, timm_network: str = "vit_base_patch32_224",
                 image_size: tuple[int, int] = (224, 224),
                 nchannels: int = 3,
                 pretrained: bool = True,
                 num_classes: int = 0,
                 ):
        super().__init__()

        model_creator = {'model_name': timm_network,
                         "pretrained": pretrained,
                         "num_classes": num_classes}

        self.encoder = timm.create_model(**model_creator)

        self.dimensionality_reductor = None

        for param in self.encoder.parameters():
            param.requires_grad = True

        n_out = get_n_out_features(self.encoder, image_size, nchannels)

        self.dimensionality_up_sampling = nn.Sequential(
            nn.Linear(n_out, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 2048), nn.ReLU(inplace=True),
            nn.Linear(2048, 4096)
        )

    def forward(self, signal_input):
        output = self.encoder(signal_input)
        output = self.dimensionality_up_sampling(output)
        return output
