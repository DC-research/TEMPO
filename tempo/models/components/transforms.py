import torch
import torch.nn as nn
import torch.distributions as D
from torch.nn import functional as F, init
from LiLY.modules import components
from LiLY.modules.components.spline import _monotonic_rational_spline
from LiLY.modules.components.linear import LULinear

from typing import (Tuple,
                    Union,
                    Optional,
                    Any)
from . import utils
from .base import (GroupLinearLayer,
                   FlowSequential)
import copy
import ipdb as pdb
# Invertible Component-wise Spline Transformation #
class ComponentWiseSpline(components.Transform):
    def __init__(
        self, 
        input_dim: int, 
        count_bins: int = 8, 
        bound: int = 3., 
        order: str = 'linear') -> None:
        """Component-wise Spline Flow
        Args:
            input_dim: The size of input/latent features.
            count_bins: The number of bins that each can have their own weights.
            bound: Tail bound (outside tail bounds the transformation is identity)
            order: Spline order

        Modified from Neural Spline Flows: https://arxiv.org/pdf/1906.04032.pdf
        """
        super(ComponentWiseSpline, self).__init__()
        assert order in ("linear", "quadratic")
        self.input_dim = input_dim
        self.count_bins = count_bins
        self.bound = bound
        self.order = order
        self.unnormalized_widths = nn.Parameter(torch.randn(self.input_dim, self.count_bins))
        self.unnormalized_heights = nn.Parameter(torch.randn(self.input_dim, self.count_bins))
        self.unnormalized_derivatives = nn.Parameter(torch.randn(self.input_dim, self.count_bins - 1))
        # Rational linear splines have additional lambda parameters
        if self.order == "linear":
            self.unnormalized_lambdas = nn.Parameter(torch.rand(self.input_dim, self.count_bins))
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_dim))
        self.register_buffer('base_dist_var', torch.eye(input_dim))

    @property
    def base_dist(self):
        return D.MultivariateNormal(self.base_dist_mean, self.base_dist_var)

    def forward(
        self, 
        x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """f: data x -> latent u"""
        u, log_detJ = self.spline_op(x)
        log_detJ = torch.sum(log_detJ, dim=1)
        return u, log_detJ

    def inverse(
        self, 
        u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """g: latent u > data x"""
        x, log_detJ = self.spline_op(u, inverse=True)
        log_detJ = torch.sum(log_detJ, dim=1)
        return x, log_detJ

    def spline_op(
        self, 
        x: torch.Tensor, 
        **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fit N separate splines for each dimension of input"""
        # widths, unnormalized_widths ~ (input_dim, num_bins)
        w = F.softmax(self.unnormalized_widths, dim=-1)
        h = F.softmax(self.unnormalized_heights, dim=-1)
        d = F.softplus(self.unnormalized_derivatives)
        if self.order == 'linear':
            l = torch.sigmoid(self.unnormalized_lambdas)
        else:
            l = None
        y, log_detJ = _monotonic_rational_spline(x, w, h, d, l, bound=self.bound, **kwargs)
        return y, log_detJ

    def log_prob(self, x):
        z, log_detJ = self.forward(x)
        logp = self.base_dist.log_prob(z) + log_detJ
        return logp
        
# Multichannel Blind Deconvolution (MBD) to recover noise epsilon_t # 

class AffineMBD(components.Transform):
    """Multichannel Blind Deconvolution (MBD) to recover noise at once.
                \epsilon_t = \sum_{p=0}^L W_l @ x_{t-k}
    """
    def __init__(
        self,
        input_size: int,
        lags: int,
        diagonal: bool,
        hidden = None) -> None:
        """Constructs MBD object

        Args:
            input_size: The number of latent causal factors.
            lags: Past time lags to consider for MBD.
            diagonal: Constrain the transition to be diagonal.
            hidden: Use hidden to constrain the rank of transition matrix.
        """
        super(AffineMBD, self).__init__()
        self.D = input_size
        self.L = lags 
        self.step_func = AffineMBDStep(self.D, self.D, self.L, diagonal, hidden)

    def forward(
        self,
        x: torch.Tensor):
        """Recover noise vector at once"""
        # x: [BS, T, D] -> [BS, T-L, L+1, D]
        x = x.unfold(dimension = 1, size = self.L+1, step = 1)
        x = torch.swapaxes(x, 2, 3)
        shape = x.shape
        x = x.reshape(-1, self.L+1, self.D)
        xx, yy = x[:,-1:], x[:,:-1]
        u, logabsdet = self.step_func(xx, yy)
        # u: [BS, T-L, D], logabsdet: [BS*T-L]
        u = u.reshape(shape[0], shape[1], self.D)
        log_detJ = torch.sum(logabsdet.reshape(shape[0], shape[1]), dim=1)
        return u, log_detJ
    
    def inverse(
        self, 
        u: torch.Tensor,
        y: torch.Tensor):
        """Generate observations/causal factors"""
        # u: [BS, T-L, D], y: [BS, L, D]
        shape = u.shape
        x = [ ]
        log_detJ = 0
        for t in range(shape[1]):
            uu = u[:,t,:]
            xx, logabsdet = self.step_func.inverse(uu, y)
            x.append(xx)
            log_detJ += logabsdet
            y = torch.cat((y[:,1:,:], xx), dim=1)
        x = torch.cat(x, dim=1)
        return x, log_detJ

class AffineMBDStep(components.Transform):
    """Multichannel Blind Deconvolution (MBD) to recover noise for one time step.
                \epsilon_t = \sum_{p=0}^L W_l @ x_{t-k}
    """
    def __init__(
        self,
        din: int,
        dout: int,
        lags: int,
        diagonal: bool,
        hidden = None) -> None:
        """Constructs MBD object

        Args:
            input_size: The number of latent causal factors.
            lags: Past time lags to consider for MBD.
        """
        super(AffineMBDStep, self).__init__()
        self.L = lags
        self.wt_func = GroupLinearLayer(din = din, 
                                        dout = dout, 
                                        num_blocks = self.L,
                                        diagonal = diagonal,
                                        hidden = hidden)
        self.b = nn.Parameter(0.01 * torch.randn(1, dout))
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply MBD for one time step"""
        # x: [BS, 1, D] y: [BS, time_lags, D] u: [BS, D]
        ut = self.wt_func(y)
        ut = torch.sum(ut, dim=1) + self.b
        # u0, logabsdet = self.w0_func(x[:,0,:])
        u0, logabsdet = x[:,0,:], torch.zeros(x.shape[0]).to(ut.device)
        u = ut + u0
        return u, logabsdet
    
    def inverse(
        self,
        u: torch.Tensor,
        y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """u0 = u - ut"""
        # u: [BS, D] y: [BS, time_lags, D] x: [BS, 1, D]
        ut = self.wt_func(y)
        ut = torch.sum(ut, dim=1)
        u0 = u - ut
        x, logabsdet = u0, torch.zeros(u.shape[0]).to(u0.device)
        x = x.unsqueeze(1)
        return x, logabsdet

# Affline Coupling Transformation #

class LinearMaskedCoupling(components.Transform):
    """ Modified RealNVP Coupling Layers per the MAF paper """
    def __init__(self, input_size, hidden_size, n_hidden, mask, cond_label_size=None):
        super().__init__()

        self.register_buffer('mask', mask)

        # scale function
        s_net = [nn.Linear(input_size + (cond_label_size if cond_label_size is not None else 0), hidden_size)]
        for _ in range(n_hidden):
            s_net += [nn.Tanh(), nn.Linear(hidden_size, hidden_size)]
        s_net += [nn.Tanh(), nn.Linear(hidden_size, input_size)]
        self.s_net = nn.Sequential(*s_net)

        # translation function
        self.t_net = copy.deepcopy(self.s_net)
        # replace Tanh with ReLU's per MAF paper
        for i in range(len(self.t_net)):
            if not isinstance(self.t_net[i], nn.Linear): self.t_net[i] = nn.ReLU()

    def forward(self, x, y=None):
        # apply mask
        mx = x * self.mask

        # run through model
        s = self.s_net(mx if y is None else torch.cat([y, mx], dim=1))
        t = self.t_net(mx if y is None else torch.cat([y, mx], dim=1))
        u = mx + (1 - self.mask) * (x - t) * torch.exp(-s)  # cf RealNVP eq 8 where u corresponds to x (here we're modeling u)

        log_abs_det_jacobian = - (1 - self.mask) * s  # log det du/dx; cf RealNVP 8 and 6; note, sum over input_size done at model log_prob
        log_abs_det_jacobian = torch.sum(log_abs_det_jacobian, dim=1)
        return u, log_abs_det_jacobian

    def inverse(self, u, y=None):
        # apply mask
        mu = u * self.mask

        # run through model
        s = self.s_net(mu if y is None else torch.cat([y, mu], dim=1))
        t = self.t_net(mu if y is None else torch.cat([y, mu], dim=1))
        x = mu + (1 - self.mask) * (u * s.exp() + t)  # cf RealNVP eq 7

        log_abs_det_jacobian = (1 - self.mask) * s  # log det dx/du
        log_abs_det_jacobian = torch.sum(log_abs_det_jacobian, dim=1)

        return x, log_abs_det_jacobian


class BatchNorm(components.Transform):
    """ RealNVP BatchNorm layer """
    def __init__(self, input_size, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))

        self.register_buffer('running_mean', torch.zeros(input_size))
        self.register_buffer('running_var', torch.ones(input_size))

    def forward(self, x, cond_y=None):
        if self.training:
            self.batch_mean = x.mean(0)
            self.batch_var = x.var(0) # note MAF paper uses biased variance estimate; ie x.var(0, unbiased=False)

            # update running mean
            self.running_mean.mul_(self.momentum).add_(self.batch_mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(self.batch_var.data * (1 - self.momentum))

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        # compute normalized input (cf original batch norm paper algo 1)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.log_gamma.exp() * x_hat + self.beta

        # compute log_abs_det_jacobian (cf RealNVP paper)
        log_abs_det_jacobian = self.log_gamma - 0.5 * torch.log(var + self.eps)
#        print('in sum log var {:6.3f} ; out sum log var {:6.3f}; sum log det {:8.3f}; mean log_gamma {:5.3f}; mean beta {:5.3f}'.format(
#            (var + self.eps).log().sum().data.numpy(), y.var(0).log().sum().data.numpy(), log_abs_det_jacobian.mean(0).item(), self.log_gamma.mean(), self.beta.mean()))
        return y, torch.sum(log_abs_det_jacobian.expand_as(x), dim=1)

    def inverse(self, y, cond_y=None):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (y - self.beta) * torch.exp(-self.log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean

        log_abs_det_jacobian = 0.5 * torch.log(var + self.eps) - self.log_gamma

        return x, torch.sum(log_abs_det_jacobian.expand_as(x), dim=1)

class AfflineCoupling(components.Transform):
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, cond_label_size=None, batch_norm=True):
        super().__init__()
        # construct model
        modules = []
        mask = torch.arange(input_size).float() % 2
        for i in range(n_blocks):
            modules += [LinearMaskedCoupling(input_size, hidden_size, n_hidden, mask, cond_label_size)]
            mask = 1 - mask
            modules += batch_norm * [BatchNorm(input_size)]

        self.net = FlowSequential(*modules)

    def forward(self, x, y=None):
        return self.net(x, y)

    def inverse(self, z, y=None):
        return self.net.inverse(z, y)