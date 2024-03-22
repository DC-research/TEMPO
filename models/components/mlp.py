import torch
import torch.nn as nn
from torch.nn import functional as F

class NLayerLeakyMLP(nn.Module):

    def __init__(self, in_features, out_features, num_layers, hidden_dim=64, bias=True):
        super().__init__()
        layers = [ ]
        for l in range(num_layers):
            if l == 0:
                layers.append(nn.Linear(in_features, hidden_dim))
                # layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.LeakyReLU(0.2))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                # layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(hidden_dim, out_features))

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class MLPEncoder(nn.Module):

    def __init__(self, latent_size, num_layers=4, hidden_dim=64):
        super().__init__()
        self.net = NLayerLeakyMLP(in_features=latent_size, 
                                  out_features=latent_size, 
                                  num_layers=num_layers, 
                                  hidden_dim=hidden_dim)
    
    def forward(self, x):
        return self.net(x)

class MLPDecoder(nn.Module):
    """Ground-truth MLP decoder used for data generation"""
    def __init__(self, latent_size, num_layers=4, hidden_dim=64):
        super().__init__()
        # TODO: Do not use ground-truth decoder architecture 
        self.net = NLayerLeakyMLP(in_features=latent_size, 
                                  out_features=latent_size, 
                                  num_layers=num_layers, 
                                  hidden_dim=hidden_dim)
    
    def forward(self, z):
        return self.net(z)

class Inference(nn.Module):
    """Ground-truth MLP decoder used for data generation"""
    def __init__(self, lag, z_dim, num_layers=4, hidden_dim=64):
        super().__init__()
        self.z_dim = z_dim
        self.lag = lag
        self.f1 = nn.Linear(lag*z_dim, z_dim*2)
        self.f2 = nn.Linear(2*hidden_dim, hidden_dim)

        self.net = NLayerLeakyMLP(in_features=hidden_dim, 
                                  out_features=z_dim*2, 
                                  num_layers=num_layers, 
                                  hidden_dim=hidden_dim)

    def forward(self, x):
        zs = x[:,:self.lag*self.z_dim]
        distributions = self.f1(zs)
        enc = self.f2(x[:,self.lag*self.z_dim:])
        distributions = distributions + self.net(enc)
        return distributions

class NAC(nn.Module):    
    def __init__(self, n_in, n_out):
        super().__init__()
        self.W_hat = nn.Parameter(torch.Tensor(n_out, n_in))
        self.M_hat = nn.Parameter(torch.Tensor(n_out, n_in))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W_hat)         
        nn.init.kaiming_uniform_(self.M_hat)

    def forward(self, input):
        weights = torch.tanh(self.W_hat) * torch.sigmoid(self.M_hat)
        return F.linear(input, weights)

class NALU(nn.Module):    
    def __init__(self, n_in, n_out):
        super().__init__()        
        self.NAC = NAC(n_in, n_out)        
        self.G = nn.Parameter(torch.Tensor(1, n_in))        
        self.eps = 1e-6        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.G)
    
    def forward(self, input):
        g = torch.sigmoid(F.linear(input, self.G))
        y1 = g * self.NAC(input)        
        y2 = (1 - g) * torch.exp(self.NAC(torch.log(torch.abs(input) + self.eps)))
        return y1 + y2

class NLayerLeakyNAC(nn.Module):

    def __init__(self, in_features, out_features, num_layers, hidden_dim=64, bias=True):
        super().__init__()
        layers = [ ]
        for l in range(num_layers):
            if l == 0:
                layers.append(NALU(in_features, hidden_dim))
                # layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.LeakyReLU(0.2))
            else:
                layers.append(NALU(hidden_dim, hidden_dim))
                # layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.LeakyReLU(0.2))
        layers.append(NALU(hidden_dim, out_features))

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

if __name__ == '__main__':
    net = NLayerLeakyMLP(3,64,32,128)
    print(net)