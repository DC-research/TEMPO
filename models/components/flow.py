import torch
import torch.nn as nn
import torch.distributions as D
from torch.nn import functional as F
from .spline import _monotonic_rational_spline, _construct_nn
import ipdb as pdb

class ComponentWiseCondSpline(nn.Module):
    def __init__(
        self, 
        input_dim, 
        context_dim=5,
        count_bins=8, 
        bound=3., 
        order='linear'):
        """Component-wise Spline Flow
        Args:
            input_dim: The size of input/latent features.
            context_dim: The size of conditioned/context features.
            count_bins: The number of bins that each can have their own weights.
            bound: Tail bound (outside tail bounds the transformation is identity)
            order: Spline order

        Modified from Neural Spline Flows: https://arxiv.org/pdf/1906.04032.pdf
        """
        super(ComponentWiseCondSpline, self).__init__()
        assert order in ("linear", "quadratic")
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.count_bins = count_bins
        self.bound = bound
        self.order = order
        self.nn = _construct_nn(input_dim=input_dim,
                                context_dim=context_dim,
                                hidden_dims=None,
                                count_bins=count_bins,
                                bound=bound,
                                order=order)
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_dim))
        self.register_buffer('base_dist_var', torch.eye(input_dim))

    @property
    def base_dist(self):
        return D.MultivariateNormal(self.base_dist_mean, self.base_dist_var)

    def _params(self, context):
        # Rational linear splines have additional lambda parameters
        if self.order == "linear":
            w, h, d, l = self.nn(context)
            # AutoRegressiveNN and DenseNN return different shapes...
            if w.shape[-1] == self.input_dim:
                l = l.transpose(-1, -2)
            else:
                l = l.reshape(l.shape[:-1] + (self.input_dim, self.count_bins))

            l = torch.sigmoid(l)

        elif self.order == "quadratic":
            w, h, d = self.nn(context)
            l = None

        # AutoRegressiveNN and DenseNN return different shapes...
        if w.shape[-1] == self.input_dim:
            w = w.transpose(-1, -2)
            h = h.transpose(-1, -2)
            d = d.transpose(-1, -2)

        else:
            w = w.reshape(w.shape[:-1] + (self.input_dim, self.count_bins))
            h = h.reshape(h.shape[:-1] + (self.input_dim, self.count_bins))
            d = d.reshape(d.shape[:-1] + (self.input_dim, self.count_bins - 1))

        w = F.softmax(w, dim=-1)
        h = F.softmax(h, dim=-1)
        d = F.softplus(d)
        return w, h, d, l
        
    def forward(self, x, context):
        """f: data x, context -> latent u"""
        u, log_detJ = self.spline_op(x, context)
        log_detJ = torch.sum(log_detJ, dim=1)
        return u, log_detJ

    def inverse(self, u, context):
        """g: latent u > data x"""
        x, log_detJ = self.spline_op(u, context, inverse=True)
        log_detJ = torch.sum(log_detJ, dim=1)
        return x, log_detJ

    def spline_op(self, x, context, **kwargs):
        """Fit N separate splines for each dimension of input"""
        # widths, unnormalized_widths ~ (input_dim, num_bins)
        w, h, d, l = self._params(context)
        y, log_detJ = _monotonic_rational_spline(x, w, h, d, l, bound=self.bound, **kwargs)
        return y, log_detJ

    def log_prob(self, x, context):
        z, log_detJ = self.forward(x, context)
        logp = self.base_dist.log_prob(z) + log_detJ
        return logp
    
    def sample(self, context, batch_size): 
        z = self.base_dist.sample((batch_size, ))
        x, _ = self.inverse(z, context)
        return x

class ComponentWiseSpline(nn.Module):
    def __init__(
        self, 
        input_dim, 
        count_bins, 
        bound=3., 
        order='linear'):
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

    def forward(self, x):
        """f: data x -> latent u"""
        u, log_detJ = self.spline_op(x)
        log_detJ = torch.sum(log_detJ, dim=1)
        return u, log_detJ

    def inverse(self, u):
        """g: latent u > data x"""
        x, log_detJ = self.spline_op(u, inverse=True)
        log_detJ = torch.sum(log_detJ, dim=1)
        return x, log_detJ

    def spline_op(self, x, **kwargs):
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

    def sample(self, batch_size): 
        z = self.base_dist.sample((batch_size, ))
        x, _ = self.inverse(z)
        return x