"""model.py"""

import torch
import ipdb as pdb
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
from .keypoint import SpatialSoftmax
import ipdb as pdb

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class BetaVAE_CNN(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, z_dim=10, nc=3, hidden_dim=256):
        super(BetaVAE_CNN, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 64, 64
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, hidden_dim, 4, 1),            # B, hidden_dim,  1,  1
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            View((-1, hidden_dim*1*1)),                 # B, hidden_dim
            nn.Linear(hidden_dim, z_dim*2),             # B, z_dim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),               # B, hidden_dim
            View((-1, hidden_dim, 1, 1)),               # B, hidden_dim,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim, 64, 4),      # B,  64,  4,  4
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 64, 64
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 128, 128
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, return_z=True):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        logvar = logvar - 2
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        if return_z:
            return x_recon, mu, logvar, z
        else:
            return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

class BetaVAE_Physics(nn.Module):
    """Visual Encoder/Decoder for Ball dataset."""
    def __init__(self, z_dim=10, nc=3, nf=16, norm_layer='Batch', hidden_dim=512):
        super(BetaVAE_Physics, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        k = 5
        self.k = k
        height = 64
        width = 64
        lim=[-5., 5., -5., 5.]
        self.height = height
        self.width = width
        self.lim = lim
        x = np.linspace(lim[0], lim[1], width // 4)
        y = np.linspace(lim[2], lim[3], height // 4)
        z = np.linspace(-1., 1., k)
        self.register_buffer('x', torch.FloatTensor(x))
        self.register_buffer('y', torch.FloatTensor(y))
        self.register_buffer('z', torch.FloatTensor(z))

        self.integrater = SpatialSoftmax(height=height//4, width=width//4, channel=k, lim=lim)
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, nf, 7, 1, 3),
            nn.BatchNorm2d(nf) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf) x 64 x 64
            nn.Conv2d(nf, nf, 5, 1, 2),
            nn.BatchNorm2d(nf) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf) x 64 x 64
            nn.Conv2d(nf, nf * 2, 4, 2, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 2) x 32 x 32
            nn.Conv2d(nf * 2, nf * 2, 3, 1, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 2) x 32 x 32
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1),
            nn.BatchNorm2d(nf * 4) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 4) x 16 x 16
            nn.Conv2d(nf * 4, k, 1, 1)
        )

        self.fc = nn.Sequential(View((-1, k * 16 * 16)),
                                nn.Linear(k * 16 * 16, z_dim)
                                )

        self.decoder = nn.Sequential(            
            nn.ConvTranspose2d(self.k, nf * 4, 4, 2, 1),
            nn.BatchNorm2d(nf * 4) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nf * 4) x 32 x 32
            nn.Conv2d(nf * 4, nf * 2, 3, 1, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nf * 4) x 32 x 32
            nn.ConvTranspose2d(nf * 2, nf * 2, 4, 2, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nf * 2) x 64 x 64
            nn.Conv2d(nf * 2, nf, 5, 1, 2),
            nn.BatchNorm2d(nf) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nf * 2) x 64 x 64
            nn.Conv2d(nf, 3, 7, 1, 3))

        # self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, return_z=True):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        if return_z:
            return x_recon, mu, logvar, z
        else:
            return x_recon, mu, logvar

    def keypoint_to_heatmap(self, keypoint, inv_std=10.):
        # keypoint: B x n_kp x 2
        # heatpmap: B x n_kp x (H / 4) x (W / 4)
        # ret: B x n_kp x (H / 4) x (W / 4)
        height = self.height // 4
        width = self.width // 4

        mu_x, mu_y = keypoint[:, :, :1].unsqueeze(-1), keypoint[:, :, 1:].unsqueeze(-1)
        y = self.y.view(1, 1, height, 1)
        x = self.x.view(1, 1, 1, width)

        g_y = (y - mu_y)**2
        g_x = (x - mu_x)**2
        dist = (g_y + g_x) * inv_std**2

        hmap = torch.exp(-dist)

        return hmap

    def _encode(self, x):
        heatmap = self.encoder(x)
        batch_size = heatmap.shape[0]
        mu = self.integrater(heatmap)
        mu = mu.view(batch_size, -1)
        logvar = self.fc(heatmap)
        return torch.cat((mu,logvar), dim=-1)

    def _decode(self, z):
        kpts = z.view(-1, self.k, 2)
        hmap = self.keypoint_to_heatmap(kpts)
        return self.decoder(hmap)

class BetaVAE_MLP(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, input_dim=3, z_dim=10, hidden_dim=128):
        super(BetaVAE_MLP, self).__init__()
        self.z_dim = z_dim
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
                                       nn.Linear(input_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, 2*z_dim)
                                    )
        # Fix the functional form to ground-truth mixing function
        self.decoder = nn.Sequential(  nn.LeakyReLU(0.2),
                                       nn.Linear(z_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.LeakyReLU(0.2),
                                       nn.Linear(hidden_dim, input_dim)
                                    )


        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, return_z=True):

        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        if return_z:
            return x_recon, mu, logvar, z
        else:
            return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()
