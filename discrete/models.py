# @Author  : Edlison
# @Date    : 5/11/21 19:40
import torch
import torch.nn as nn
from v6.discrete.utils import initialize_weights


class Generator(nn.Module):
    def __init__(self, latent_dim, cls_num, features_dim, discrete=False):
        super().__init__()
        self.name = 'Generator'
        self.discrete = discrete
        if discrete:
            features_dim = features_dim * 2
        self.net = nn.Sequential(
            nn.Linear(latent_dim + cls_num, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim)
        )
        initialize_weights(self)

    def forward(self, zn, zc):
        gen = self.net(torch.cat([zn, zc], dim=1))
        if self.discrete:
            features_split = torch.split(gen, 2, dim=1)
            res = []
            for feature in features_split:
                res.append(torch.softmax(feature, dim=1))
            gen = torch.cat(res, dim=1)
        else:
            gen = torch.tanh(gen)
        return gen


class Discriminator(nn.Module):
    def __init__(self, features_dim, discrete=False):
        super().__init__()
        self.name = 'Discriminator'
        if discrete:
            features_dim = features_dim * 2
        self.net = nn.Sequential(
            nn.Linear(features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        initialize_weights(self)

    def forward(self, x):
        out = self.net(x)
        return out


class Encoder(nn.Module):
    def __init__(self, latent_dim, cls_num, features_dim, discrete=False):
        super().__init__()
        self.name = 'Encoder'
        self.latent_dim = latent_dim
        if discrete:
            features_dim = features_dim * 2
        self.net = nn.Sequential(
            nn.Linear(features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim + cls_num)
        )
        initialize_weights(self)

    def forward(self, x):
        z = self.net(x)
        zn = z[:, :self.latent_dim]
        zc_logits = z[:, self.latent_dim:]
        zc = torch.softmax(zc_logits, dim=1)
        return zn, zc, zc_logits
