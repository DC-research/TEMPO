"""Prior Network"""
import torch
import torch.nn as nn
from torch.autograd.functional import jacobian
from .mlp import NLayerLeakyMLP, NLayerLeakyNAC
from .base import GroupLinearLayer
import ipdb as pdb


class MBDTransitionPrior(nn.Module):

    def __init__(self, lags, latent_size, bias=False):
        super().__init__()
        # self.init_hiddens = nn.Parameter(0.001 * torch.randn(lags, latent_size))    
        # out[:,:,0] = (x[:,:,0]@conv.weight[:,:,0].T)+(x[:,:,1]@conv.weight[:,:,1].T) 
        # out[:,:,1] = (x[:,:,1]@conv.weight[:,:,0].T)+(x[:,:,2]@conv.weight[:,:,1].T)
        self.L = lags      
        self.transition = GroupLinearLayer(din = latent_size, 
                                           dout = latent_size, 
                                           num_blocks = lags,
                                           diagonal = False)
        self.bias = bias
        if bias:
            self.b = nn.Parameter(0.001 * torch.randn(1, latent_size))
    
    def forward(self, x, mask=None):
        # x: [BS, T, D] -> [BS, T-L, L+1, D]
        batch_size, length, input_dim = x.shape
        # init_hiddens = self.init_hiddens.repeat(batch_size, 1, 1)
        # x = torch.cat((init_hiddens, x), dim=1)
        x = x.unfold(dimension = 1, size = self.L+1, step = 1)
        x = torch.swapaxes(x, 2, 3)
        shape = x.shape

        x = x.reshape(-1, self.L+1, input_dim)
        xx, yy = x[:,-1:], x[:,:-1]
        if self.bias:
            residuals = torch.sum(self.transition(yy), dim=1) + self.b - xx.squeeze()
        else:
            residuals = torch.sum(self.transition(yy), dim=1) - xx.squeeze()
        residuals = residuals.reshape(batch_size, -1, input_dim)
        # Dummy jacobian matrix (0) to represent identity mapping
        log_abs_det_jacobian = torch.zeros(batch_size, device=x.device)
        return residuals, log_abs_det_jacobian

class NPTransitionPrior(nn.Module):

    def __init__(
        self, 
        lags, 
        latent_size, 
        num_layers=3, 
        hidden_dim=64):
        super().__init__()
        self.L = lags
        # self.init_hiddens = nn.Parameter(0.01 * torch.randn(lags, latent_size))       
        gs = [NLayerLeakyMLP(in_features=lags*latent_size+1, 
                             out_features=1, 
                             num_layers=num_layers, 
                             hidden_dim=hidden_dim) for i in range(latent_size)]
        
        self.gs = nn.ModuleList(gs)
    
    def forward(self, x, masks=None):
        # x: [BS, T, D] -> [BS, T-L, L+1, D]
        batch_size, length, input_dim = x.shape
        # init_hiddens = self.init_hiddens.repeat(batch_size, 1, 1)
        # x = torch.cat((init_hiddens, x), dim=1)
        x = x.unfold(dimension = 1, size = self.L+1, step = 1)
        x = torch.swapaxes(x, 2, 3)
        shape = x.shape
        x = x.reshape(-1, self.L+1, input_dim)
        xx, yy = x[:,-1:], x[:,:-1]
        yy = yy.reshape(-1, self.L*input_dim)
        residuals = [ ]
        sum_log_abs_det_jacobian = 0
        for i in range(input_dim):
            if masks is None:
                inputs = torch.cat((yy, xx[:,:,i]),dim=-1)
            else:
                mask = masks[i]
                inputs = torch.cat((yy*mask, xx[:,:,i]),dim=-1)
            residual = self.gs[i](inputs)
            with torch.enable_grad():
                pdd = jacobian(self.gs[i], inputs, create_graph=True, vectorize=True)
            # Determinant of low-triangular mat is product of diagonal entries
            logabsdet = torch.log(torch.abs(torch.diag(pdd[:,0,:,-1])))
            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)

        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, -1, input_dim)
        sum_log_abs_det_jacobian = torch.sum(sum_log_abs_det_jacobian.reshape(batch_size, length-self.L), dim=1)
        return residuals, sum_log_abs_det_jacobian

class NPChangeTransitionPrior(nn.Module):

    def __init__(
        self, 
        lags, 
        latent_size,
        embedding_dim, 
        num_layers=3,
        hidden_dim=64):
        super().__init__()
        self.L = lags
        # self.init_hiddens = nn.Parameter(0.01 * torch.randn(lags, latent_size))       
        gs = [NLayerLeakyMLP(in_features=hidden_dim+lags*latent_size+1, 
                             out_features=1, 
                             num_layers=num_layers, 
                             hidden_dim=hidden_dim) for i in range(latent_size)]
        
        self.gs = nn.ModuleList(gs)
        self.fc = NLayerLeakyMLP(in_features=embedding_dim,
                                 out_features=hidden_dim,
                                 num_layers=2,
                                 hidden_dim=hidden_dim)

    def forward(self, x, embeddings, masks=None):
        # x: [BS, T, D] -> [BS, T-L, L+1, D]
        # embeddings: [BS, embed_dims]
        batch_size, length, input_dim = x.shape
        embeddings = self.fc(embeddings)
        # init_hiddens = self.init_hiddens.repeat(batch_size, 1, 1)
        # x = torch.cat((init_hiddens, x), dim=1)
        x = x.unfold(dimension = 1, size = self.L+1, step = 1)
        x = torch.swapaxes(x, 2, 3)
        shape = x.shape
        x = x.reshape(-1, self.L+1, input_dim)
        xx, yy = x[:,-1:], x[:,:-1]
        yy = yy.reshape(-1, self.L*input_dim)
        residuals = [ ]
        sum_log_abs_det_jacobian = 0
        for i in range(input_dim):
            if masks is None:
                inputs = torch.cat((embeddings, yy, xx[:,:,i]),dim=-1)
            else:
                mask = masks[i]
                inputs = torch.cat((embeddings, yy*mask, xx[:,:,i]),dim=-1)
            residual = self.gs[i](inputs)
            with torch.enable_grad():
                pdd = jacobian(self.gs[i], inputs, create_graph=True, vectorize=True)
            # Determinant of low-triangular mat is product of diagonal entries
            logabsdet = torch.log(torch.abs(torch.diag(pdd[:,0,:,-1])))
            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)

        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, -1, input_dim)
        sum_log_abs_det_jacobian = torch.sum(sum_log_abs_det_jacobian.reshape(batch_size, length-self.L), dim=1)
        return residuals, sum_log_abs_det_jacobian
