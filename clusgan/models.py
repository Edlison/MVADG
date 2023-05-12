import numpy as np
import torch.nn as nn
import torch
from .utils import tlog, softmax, initialize_weights, calc_gradient_penalty
import torch.nn.functional as F

torch.backends.cudnn.enabled = False  # 关闭cudnn


class Reshape(nn.Module):
    def __init__(self, shape=[]):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)

    def extra_repr(self):
        return 'shape={}'.format(
            self.shape
        )


class Generator_CNN(nn.Module):
    def __init__(self, latent_dim, n_c, x_shape):
        super(Generator_CNN, self).__init__()
        self.name = 'generator'
        self.latent_dim = latent_dim
        self.n_c = n_c
        self.x_shape = x_shape
        self.ishape = (128, 7, 7)
        self.iels = int(np.prod(self.ishape))
        self.model = nn.Sequential(
            torch.nn.Linear(self.latent_dim + self.n_c, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(1024, self.iels),
            nn.BatchNorm1d(self.iels),
            nn.LeakyReLU(0.2, inplace=True),
            Reshape(self.ishape),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, bias=True),
            nn.Sigmoid()
        )
        initialize_weights(self)

    def forward(self, zn, zc):
        z = torch.cat((zn, zc), 1)
        x_gen = self.model(z)
        x_gen = x_gen.view(x_gen.size(0), *self.x_shape)
        return x_gen


class Encoder_CNN(nn.Module):
    def __init__(self, latent_dim, n_c):
        super(Encoder_CNN, self).__init__()
        self.name = 'encoder'
        self.channels = 1
        self.latent_dim = latent_dim
        self.n_c = n_c
        self.cshape = (128, 5, 5)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.model = nn.Sequential(
            nn.Conv2d(self.channels, 64, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            Reshape(self.lshape),
            torch.nn.Linear(self.iels, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(1024, latent_dim + n_c)
        )
        initialize_weights(self)

    def forward(self, in_feat):
        z_img = self.model(in_feat)
        z = z_img.view(z_img.shape[0], -1)
        zn = z[:, 0:self.latent_dim]
        zc_logits = z[:, self.latent_dim:]
        zc = softmax(zc_logits)
        return zn, zc, zc_logits


class Discriminator_CNN(nn.Module):
    def __init__(self, wass_metric=True):
        super(Discriminator_CNN, self).__init__()
        self.name = 'discriminator'
        self.channels = 1
        self.cshape = (128, 5, 5)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.wass = wass_metric
        self.model = nn.Sequential(
            nn.Conv2d(self.channels, 64, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            Reshape(self.lshape),
            torch.nn.Linear(self.iels, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(1024, 1),
        )

        if (not self.wass):
            self.model = nn.Sequential(self.model, torch.nn.Sigmoid())
        initialize_weights(self)

    def forward(self, img):
        validity = self.model(img)
        return validity


class Generator_Usps(nn.Module):
    def __init__(self, latent_dim, n_c, x_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_c = n_c
        self.x_shape = x_shape
        self.ishape = (128, 4, 4)
        self.iels = int(np.prod(self.ishape))
        self.model = nn.Sequential(
            torch.nn.Linear(self.latent_dim + self.n_c, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(1024, self.iels),
            nn.BatchNorm1d(self.iels),
            nn.LeakyReLU(0.2, inplace=True),
            Reshape(self.ishape),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True),
            # out = (7 - 1) * 2 - 2 * 1 + 4 = 14
            # out = (4 - 1) * 2 - 2 * 1 + 4 = 8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, bias=True),
            # out = (14 - 1) * 2 - 2 * 1 + 4 = 28
            # out = (8 - 1) * 2 - 2 * 1 + 4 = 16
            nn.Sigmoid()
        )
        initialize_weights(self)

    def forward(self, zn, zc):
        z = torch.cat((zn, zc), 1)
        x_gen = self.model(z)
        x_gen = x_gen.view(x_gen.size(0), *self.x_shape)
        return x_gen


class Encoder_Usps(nn.Module):
    def __init__(self, latent_dim, n_c):
        super().__init__()
        self.channels = 1
        self.latent_dim = latent_dim
        self.n_c = n_c
        self.cshape = (128, 2, 2)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.model = nn.Sequential(
            nn.Conv2d(self.channels, 64, 4, stride=2, bias=True),
            # out = (28 - 1 * (4 - 1) - 1) / 2 + 1 = 13
            # out = (16 - 1 * (4 - 1) - 1) / 2 + 1 = 7
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, bias=True),
            # out = (13 - 1 * (4 - 1) - 1) / 2 + 1 = 5.5
            # out = (7 - 1 * (4 - 1) - 1) / 2 + 1 = 2.5
            nn.LeakyReLU(0.2, inplace=True),
            Reshape(self.lshape),
            torch.nn.Linear(self.iels, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(1024, latent_dim + n_c)
        )
        initialize_weights(self)

    def forward(self, in_feat):
        z_img = self.model(in_feat)
        z = z_img.view(z_img.shape[0], -1)
        zn = z[:, 0:self.latent_dim]
        zc_logits = z[:, self.latent_dim:]
        zc = softmax(zc_logits)
        return zn, zc, zc_logits


class Discriminator_Usps(nn.Module):
    def __init__(self, wass_metric=True):
        super().__init__()
        self.channels = 1
        self.cshape = (128, 2, 2)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.wass = wass_metric
        self.model = nn.Sequential(
            nn.Conv2d(self.channels, 64, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            Reshape(self.lshape),
            torch.nn.Linear(self.iels, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(1024, 1),
        )

        if (not self.wass):
            self.model = nn.Sequential(self.model, torch.nn.Sigmoid())
        initialize_weights(self)

    def forward(self, img):
        validity = self.model(img)
        return validity


class Generator_Simple(nn.Module):
    def __init__(self, zn=30, zc=3, image_size=16, channels=1, conv_dim=8):
        super().__init__()
        self.image_size = image_size
        self.fc1 = nn.Linear(zn + zc, (self.image_size // 4) * (self.image_size // 4) * conv_dim * 2)
        self.tconv2 = conv_block(conv_dim * 2, conv_dim, transpose=True, use_bn=True)
        self.tconv3 = conv_block(conv_dim, channels, transpose=True, use_bn=False)

    def forward(self, zn, zc):
        x = F.relu(self.fc1(torch.cat((zn, zc), dim=1)))
        x = x.reshape([x.shape[0], -1, self.image_size // 4, self.image_size // 4])
        x = F.relu(self.tconv2(x))
        x = torch.tanh(self.tconv3(x))
        return x


class Discriminator_Simple(nn.Module):
    def __init__(self, image_size=16, channels=1, conv_dim=8, wass=False):
        super().__init__()
        self.wass = wass
        self.conv1 = conv_block(channels, conv_dim, use_bn=False)
        self.conv2 = conv_block(conv_dim, conv_dim * 2, use_bn=True)
        self.fc3 = nn.Linear((image_size // 4) * (image_size // 4) * conv_dim * 2, 1)

    def forward(self, x):
        alpha = 0.2
        x = F.leaky_relu(self.conv1(x), alpha)
        x = F.leaky_relu(self.conv2(x), alpha)
        x = x.reshape([x.shape[0], -1])
        x = self.fc3(x)
        if not self.wass:
            x = torch.sigmoid(x)  # 用WGAN的话 不能有sigmoid
        return x.squeeze()


class Encoder_Simple(nn.Module):
    def __init__(self, zn=30, zc=3, image_size=16, channels=1, conv_dim=8):
        super().__init__()
        self.latent_dim = zn
        self.conv1 = conv_block(channels, conv_dim, use_bn=False)
        self.conv2 = conv_block(conv_dim, conv_dim * 2, use_bn=True)
        self.fc3 = nn.Linear((image_size // 4) * (image_size // 4) * conv_dim * 2, zn + zc)

    def forward(self, x):
        alpha = 0.2
        x = F.leaky_relu(self.conv1(x), alpha)
        x = F.leaky_relu(self.conv2(x), alpha)
        x = x.reshape([x.shape[0], -1])
        x = self.fc3(x)
        zn = x[:, :self.latent_dim]
        zc_logits = x[:, self.latent_dim:]
        zc = softmax(zc_logits)
        return zn, zc, zc_logits


def conv_block(c_in, c_out, k_size=4, stride=2, pad=1, use_bn=True, transpose=False):
    module = []
    if transpose:
        module.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=not use_bn))
    else:
        module.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=not use_bn))
    if use_bn:
        module.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*module)


class MultiNet_mnist(nn.Module):
    def __init__(self, view_num, latent_dim, cls_num, x_shape, wass=True):
        super().__init__()
        self.view_num = view_num
        self.latent_dim = latent_dim
        self.cls_num = cls_num
        self.Gs = nn.ModuleList(
            [Generator_CNN(latent_dim=latent_dim, n_c=cls_num, x_shape=x_shape) for _ in range(view_num)])
        self.Ds = nn.ModuleList([Discriminator_CNN(wass_metric=wass) for _ in range(view_num)])
        self.Es = nn.ModuleList([Encoder_CNN(latent_dim=latent_dim, n_c=cls_num) for _ in range(view_num)])

    def forward(self, zn=None, zc=None, data=None, mode='G'):
        if mode == 'G':
            g = []
            for i in range(self.view_num):
                g.append(self.Gs[i](zn[i], zc[i]))
            return g
        elif mode == 'D':
            d = []
            for i in range(self.view_num):
                d.append(self.Ds[i](data[i]))
            return d
        elif mode == 'E':
            e = []
            for i in range(self.view_num):
                e.append(self.Es[i](data[i]))
            return e
        else:
            raise ValueError('err mode')


class MultiNet_usps(nn.Module):
    def __init__(self, view_num, latent_dim, cls_num, x_shape, wass=True):
        super().__init__()
        self.view_num = view_num
        self.Gs = nn.ModuleList(
            [Generator_Usps(latent_dim=latent_dim, n_c=cls_num, x_shape=x_shape) for _ in range(view_num)])
        self.Ds = nn.ModuleList([Discriminator_Usps(wass_metric=wass) for _ in range(view_num)])
        self.Es = nn.ModuleList([Encoder_Usps(latent_dim=latent_dim, n_c=cls_num) for _ in range(view_num)])

    def forward(self, zn=None, zc=None, data=None, mode='G'):
        if mode == 'G':
            g = []
            for i in range(self.view_num):
                g.append(self.Gs[i](zn[i], zc[i]))
            return g
        elif mode == 'D':
            d = []
            for i in range(self.view_num):
                d.append(self.Ds[i](data[i]))
            return d
        elif mode == 'E':
            e = []
            for i in range(self.view_num):
                e.append(self.Es[i](data[i]))
            return e
        else:
            raise ValueError('err mode')


class MultiNet_mnist_usps(nn.Module):
    def __init__(self, view_num, latent_dim, cls_num, x_shape1, x_shape2, wass=True):
        super().__init__()
        self.view_num = view_num
        self.Gs = nn.ModuleList(
            [Generator_CNN(latent_dim=latent_dim, n_c=cls_num, x_shape=x_shape1),
             Generator_Usps(latent_dim=latent_dim, n_c=cls_num, x_shape=x_shape2)])
        self.Ds = nn.ModuleList(
            [Discriminator_CNN(wass_metric=wass), Discriminator_Usps(wass_metric=wass)])
        self.Es = nn.ModuleList(
            [Encoder_CNN(latent_dim=latent_dim, n_c=cls_num), Encoder_Usps(latent_dim=latent_dim, n_c=cls_num)])

    def forward(self, zn=None, zc=None, data=None, mode='G'):
        if mode == 'G':
            g = []
            for i in range(self.view_num):
                g.append(self.Gs[i](zn[i], zc[i]))
            return g
        elif mode == 'D':
            d = []
            for i in range(self.view_num):
                d.append(self.Ds[i](data[i]))
            return d
        elif mode == 'E':
            e = []
            for i in range(self.view_num):
                e.append(self.Es[i](data[i]))
            return e
        else:
            raise ValueError('err mode')


class MultiNet_simple(nn.Module):
    def __init__(self, view_num, latent_dim, cls_num, wass=True):
        super().__init__()
        self.view_num = view_num
        self.Gs = nn.ModuleList(
            [Generator_Simple(zn=latent_dim, zc=cls_num) for _ in range(view_num)])
        self.Ds = nn.ModuleList([Discriminator_Simple(wass=wass) for _ in range(view_num)])
        self.Es = nn.ModuleList([Encoder_Simple(zn=latent_dim, zc=cls_num) for _ in range(view_num)])

    def forward(self, zn=None, zc=None, data=None, mode='G'):
        if mode == 'G':
            g = []
            for i in range(self.view_num):
                g.append(self.Gs[i](zn[i], zc[i]))
            return g
        elif mode == 'D':
            d = []
            for i in range(self.view_num):
                d.append(self.Ds[i](data[i]))
            return d
        elif mode == 'E':
            e = []
            for i in range(self.view_num):
                e.append(self.Es[i](data[i]))
            return e
        else:
            raise ValueError('err mode')


class MultiNet_usps_simple(nn.Module):
    def __init__(self, view_num, latent_dim, cls_num, wass=True):
        super().__init__()
        x_shape = (1, 16, 16)
        self.view_num = view_num
        self.Gs = nn.ModuleList(
            [Generator_Usps(latent_dim, cls_num, x_shape=x_shape), Generator_Simple(zn=latent_dim, zc=cls_num)])
        self.Ds = nn.ModuleList([Discriminator_Usps(wass_metric=wass), Discriminator_Simple(wass=wass)])
        self.Es = nn.ModuleList([Encoder_Usps(latent_dim, cls_num), Encoder_Simple(zn=latent_dim, zc=cls_num)])

    def forward(self, zn=None, zc=None, data=None, mode='G'):
        if mode == 'G':
            g = []
            for i in range(self.view_num):
                g.append(self.Gs[i](zn[i], zc[i]))
            return g
        elif mode == 'D':
            d = []
            for i in range(self.view_num):
                d.append(self.Ds[i](data[i]))
            return d
        elif mode == 'E':
            e = []
            for i in range(self.view_num):
                e.append(self.Es[i](data[i]))
            return e
        else:
            raise ValueError('err mode')
