"""Basic definitions for the transforms module."""
"""https://github.com/bayesiains/nsf/blob/master/nde/transforms/base.py"""

import numpy as np
import torch
from torch import nn
from . import utils
from typing import (Tuple,
                    Union)

class InverseNotAvailable(Exception):
    """Exception to be thrown when a transform does not have an inverse."""
    pass


class InputOutsideDomain(Exception):
    """Exception to be thrown when the input to a transform is not within its domain."""
    pass

class Namespace(object):
    """Converts dict to class attributes as Argparse"""
    def __init__(self, **kwds):
        self.__dict__.update(kwds)
    def __repr__(self):
        items = list(self.__dict__.items())
        temp = []
        for name, value in items:
            if not name.startswith('_'):
                temp.append('%s=%r' % (name, value))
        temp.sort()
        return '%s(%s)' % (self.__class__.__name__, ', '.join(temp))

class GroupLinearLayer(nn.Module):
    """GroupLinearLayer computes N dinstinct linear transformations at once"""
    def __init__(
        self, 
        din: int, 
        dout: int, 
        num_blocks: int,
        diagonal: bool = False,
        hidden: Union[None, int] = None) -> None:
        """Group Linear Layer module

        Args:
            din: The feature dimension of input data.
            dout: The projected dimensions of data.
            num_blocks: The number of linear transformation to compute at once.
            diagonal: Whether transition matrix is diagonal
        """
        super(GroupLinearLayer, self).__init__()
        assert (hidden is None) or (type(hidden) == int)
        self.hidden = hidden
        self.diagonal = diagonal
        # Sparse transition already implements low-rank
        assert (bool(self.hidden) and self.diagonal) == False
        if diagonal:
            self.d = nn.Parameter(0.01 * torch.randn(num_blocks, dout))
        else:
            if hidden is None:
                self.w = nn.Parameter(0.01 * torch.randn(num_blocks, din, dout))
            else:
                assert isinstance(hidden, int)
                self.wh = nn.Parameter(0.01 * torch.randn(num_blocks, din, hidden))
                self.hw = nn.Parameter(0.01 * torch.randn(num_blocks, hidden, dout))

    def forward(
        self,
        x: torch.Tensor) -> torch.Tensor:
        if self.diagonal:
            w = torch.diag_embed(self.d)
            # x: [BS,num_blocks,din]->[num_blocks,BS,din]
            x = x.permute(1,0,2)
            x = torch.bmm(x, w)
            # x: [BS,num_blocks,dout]
            x = x.permute(1,0,2)
        elif self.hidden is None:
            x = x.permute(1,0,2)
            x = torch.bmm(x, self.w)
            # x: [BS,num_blocks,dout]
            x = x.permute(1,0,2)
        else:
            x = x.permute(1,0,2)
            # x: [num_blocks,BS,din]->[num_blocks,BS,hidden]
            x = torch.bmm(x, self.wh)           
            x = torch.bmm(x, self.hw)  
            x = x.permute(1,0,2)
        return x
    
    def get_weight_matrix(self):
        if self.diagonal:
            return torch.diag_embed(self.d)
        elif self.hidden is None:
            return self.w 
        else:
            return torch.matmul(self.wh, self.hw)

class Transform(nn.Module):
    """Base class for all transform objects."""

    def forward(self, inputs, context=None):
        raise NotImplementedError()

    def inverse(self, inputs, context=None):
        raise InverseNotAvailable()


class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """
    def forward(self, x, y):
        sum_log_abs_det_jacobians = 0
        for module in self:
            x, log_abs_det_jacobian = module(x, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return x, sum_log_abs_det_jacobians

    def inverse(self, u, y):
        sum_log_abs_det_jacobians = 0
        for module in reversed(self):
            u, log_abs_det_jacobian = module.inverse(u, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return u, sum_log_abs_det_jacobians

        
class CompositeTransform(Transform):
    """Composes several transforms into one, in the order they are given."""

    def __init__(self, transforms):
        """Constructor.

        Args:
            transforms: an iterable of `Transform` objects.
        """
        super().__init__()
        self._transforms = nn.ModuleList(transforms)

    @staticmethod
    def _cascade(inputs, funcs, context):
        batch_size = inputs.shape[0]
        outputs = inputs
        total_logabsdet = torch.zeros(batch_size)
        for func in funcs:
            outputs, logabsdet = func(outputs, context)
            total_logabsdet += logabsdet
        return outputs, total_logabsdet

    def forward(self, inputs, context=None):
        funcs = self._transforms
        return self._cascade(inputs, funcs, context)

    def inverse(self, inputs, context=None):
        funcs = (transform.inverse for transform in self._transforms[::-1])
        return self._cascade(inputs, funcs, context)


class MultiscaleCompositeTransform(Transform):
    """A multiscale composite transform as described in the RealNVP paper.

    Splits the outputs along the given dimension after every transform, outputs one half, and
    passes the other half to further transforms. No splitting is done before the last transform.

    Note: Inputs could be of arbitrary shape, but outputs will always be flattened.

    Reference:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.
    """

    def __init__(self, num_transforms, split_dim=1):
        """Constructor.

        Args:
            num_transforms: int, total number of transforms to be added.
            split_dim: dimension along which to split.
        """
        if not utils.is_positive_int(split_dim):
            raise TypeError('Split dimension must be a positive integer.')

        super().__init__()
        self._transforms = nn.ModuleList()
        self._output_shapes = []
        self._num_transforms = num_transforms
        self._split_dim = split_dim

    def add_transform(self, transform, transform_output_shape):
        """Add a transform. Must be called exactly `num_transforms` times.

        Parameters:
            transform: the `Transform` object to be added.
            transform_output_shape: tuple, shape of transform's outputs, excl. the first batch
                dimension.

        Returns:
            Input shape for the next transform, or None if adding the last transform.
        """
        assert len(self._transforms) <= self._num_transforms

        if len(self._transforms) == self._num_transforms:
            raise RuntimeError(
                'Adding more than {} transforms is not allowed.'.format(self._num_transforms))

        if (self._split_dim - 1) >= len(transform_output_shape):
            raise ValueError('No split_dim in output shape')

        if transform_output_shape[self._split_dim - 1] < 2:
            raise ValueError('Size of dimension {} must be at least 2.'.format(self._split_dim))

        self._transforms.append(transform)

        if len(self._transforms) != self._num_transforms:  # Unless last transform.
            output_shape = list(transform_output_shape)
            output_shape[self._split_dim - 1] = (output_shape[self._split_dim - 1] + 1) // 2
            output_shape = tuple(output_shape)

            hidden_shape = list(transform_output_shape)
            hidden_shape[self._split_dim - 1] = hidden_shape[self._split_dim - 1] // 2
            hidden_shape = tuple(hidden_shape)
        else:
            # No splitting for last transform.
            output_shape = transform_output_shape
            hidden_shape = None

        self._output_shapes.append(output_shape)
        return hidden_shape

    def forward(self, inputs, context=None):
        if self._split_dim >= inputs.dim():
            raise ValueError('No split_dim in inputs.')
        if self._num_transforms != len(self._transforms):
            raise RuntimeError('Expecting exactly {} transform(s) '
                               'to be added.'.format(self._num_transforms))

        batch_size = inputs.shape[0]

        def cascade():
            hiddens = inputs

            for i, transform in enumerate(self._transforms[:-1]):
                transform_outputs, logabsdet = transform(hiddens, context)
                outputs, hiddens = torch.chunk(transform_outputs,
                                               chunks=2,
                                               dim=self._split_dim)
                assert outputs.shape[1:] == self._output_shapes[i]
                yield outputs, logabsdet

            # Don't do the splitting for the last transform.
            outputs, logabsdet = self._transforms[-1](hiddens, context)
            yield outputs, logabsdet

        all_outputs = []
        total_logabsdet = torch.zeros(batch_size)

        for outputs, logabsdet in cascade():
            all_outputs.append(outputs.reshape(batch_size, -1))
            total_logabsdet += logabsdet

        all_outputs = torch.cat(all_outputs, dim=-1)
        return all_outputs, total_logabsdet

    def inverse(self, inputs, context=None):
        if inputs.dim() != 2:
            raise ValueError('Expecting NxD inputs')
        if self._num_transforms != len(self._transforms):
            raise RuntimeError('Expecting exactly {} transform(s) '
                               'to be added.'.format(self._num_transforms))

        batch_size = inputs.shape[0]

        rev_inv_transforms = [transform.inverse for transform in self._transforms[::-1]]

        split_indices = np.cumsum([np.prod(shape) for shape in self._output_shapes])
        split_indices = np.insert(split_indices, 0, 0)

        split_inputs = []
        for i in range(len(self._output_shapes)):
            flat_input = inputs[:, split_indices[i]:split_indices[i+1]]
            split_inputs.append(flat_input.view(-1, *self._output_shapes[i]))
        rev_split_inputs = split_inputs[::-1]

        total_logabsdet = torch.zeros(batch_size)

        # We don't do the splitting for the last (here first) transform.
        hiddens, logabsdet = rev_inv_transforms[0](rev_split_inputs[0], context)
        total_logabsdet += logabsdet

        for inv_transform, input_chunk in zip(rev_inv_transforms[1:], rev_split_inputs[1:]):
            tmp_concat_inputs = torch.cat([input_chunk, hiddens], dim=self._split_dim)
            hiddens, logabsdet = inv_transform(tmp_concat_inputs, context)
            total_logabsdet += logabsdet

        outputs = hiddens

        return outputs, total_logabsdet


class InverseTransform(Transform):
    """Creates a transform that is the inverse of a given transform."""

    def __init__(self, transform):
        """Constructor.

        Args:
            transform: An object of type `Transform`.
        """
        super().__init__()
        self._transform = transform

    def forward(self, inputs, context=None):
        return self._transform.inverse(inputs, context)

    def inverse(self, inputs, context=None):
        return self._transform(inputs, context)