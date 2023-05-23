import torch.nn as nn
import timm
import torch


def get_n_out_features(encoder, img_size, nchannels):
    out_feature = encoder(torch.randn(1, nchannels, img_size[0], img_size[1]))
    n_out = 1
    for dim in out_feature[-1].shape:
        n_out *= dim
    return n_out

class TimmSimpleCNN(nn.Module):
    def __init__(self, network: str,
                 image_size: int,
                 nchannels: int,
                 transformers: bool = False,
                 pretrained: bool = False,
                 num_classes: int = 0,
                 features_only: bool = True):

        super().__init__()
        if transformers:
            model_creator = {'model_name': network,
                             "pretrained": pretrained,
                             "num_classes": num_classes}
        else:
            model_creator = {'model_name': network,
                             "pretrained": pretrained,
                             "features_only": features_only}

        self.encoder = timm.create_model(**model_creator)

        self.dimensionality_reductor = None

        for param in self.encoder.parameters():
            param.requires_grad = True

        n_out = get_n_out_features(self.encoder, image_size, nchannels)

        if transformers:
            self.dimensionality_up_sampling = nn.Sequential(
                nn.Linear(n_out, 512), nn.ReLU(inplace=True),
                nn.Linear(512, 1024), nn.ReLU(inplace=True),
                nn.Linear(1024, 2048)
            )
        else:
            self.dimensionality_up_sampling = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n_out, 512), nn.ReLU(inplace=True),
                nn.Linear(512, 1024), nn.ReLU(inplace=True),
                nn.Linear(1024, 2048)
            )

    def forward(self, signal_input):
        # output = self.encoder(signal_input)[-1]
        output = self.encoder(signal_input)
        output = self.dimensionality_up_sampling(output)

        return output


class TimmSimpleCNNTrack3(nn.Module):
    def __init__(self, network: str,
                 image_size: int,
                 nchannels: int,
                 transformers: bool = False,
                 pretrained: bool = False,
                 num_classes: int = 0,
                 features_only: bool = True):

        super().__init__()
        if transformers:
            model_creator = {'model_name': network,
                             "pretrained": pretrained,
                             "num_classes": num_classes}
        else:
            model_creator = {'model_name': network,
                             "pretrained": pretrained,
                             "features_only": features_only}

        self.encoder = timm.create_model(**model_creator)

        self.dimensionality_reductor = None

        for param in self.encoder.parameters():
            param.requires_grad = True

        n_out = get_n_out_features(self.encoder, image_size, nchannels)

        if transformers:
            self.dimensionality_up_sampling = nn.Sequential(
                nn.Linear(n_out, 512), nn.ReLU(inplace=True),
                nn.Linear(512, 1024), nn.ReLU(inplace=True),
                nn.Linear(1024, 2048), nn.ReLU(inplace=True),
                nn.Linear(2048, 4096)
            )
        else:
            self.dimensionality_up_sampling = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n_out, 512), nn.ReLU(inplace=True),
                nn.Linear(512, 1024), nn.ReLU(inplace=True),
                nn.Linear(1024, 2048), nn.ReLU(inplace=True),
                nn.Linear(2048, 4096)
            )

    def forward(self, signal_input):
        # output = self.encoder(signal_input)[-1]
        output = self.encoder(signal_input)
        output = self.dimensionality_up_sampling(output)

        return output

