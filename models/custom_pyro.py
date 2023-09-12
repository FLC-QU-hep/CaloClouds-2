# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import partial, reduce
import operator

import torch
from torch.distributions.utils import _sum_rightmost


from pyro.nn import AutoRegressiveNN, ConditionalAutoRegressiveNN, ConditionalDenseNN, DenseNN

from pyro.distributions import constraints
from pyro.distributions.conditional import ConditionalTransformModule
from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.util import copy_docs_from
from pyro.distributions.transforms.spline import ConditionalSpline
from pyro.distributions.transforms.utils import clamp_preserve_gradients
from pyro.distributions.transforms import SplineCoupling


    



class SplineAutoregressive(TransformModule):
    r"""
    An implementation of the autoregressive layer with rational spline bijections of
    linear and quadratic order (Durkan et al., 2019; Dolatabadi et al., 2020).
    Rational splines are functions that are comprised of segments that are the ratio
    of two polynomials (see :class:`~pyro.distributions.transforms.Spline`).

    The autoregressive layer uses the transformation,

        :math:`y_d = g_{\theta_d}(x_d)\ \ \ d=1,2,\ldots,D`

    where :math:`\mathbf{x}=(x_1,x_2,\ldots,x_D)` are the inputs,
    :math:`\mathbf{y}=(y_1,y_2,\ldots,y_D)` are the outputs, :math:`g_{\theta_d}` is
    an elementwise rational monotonic spline with parameters :math:`\theta_d`, and
    :math:`\theta=(\theta_1,\theta_2,\ldots,\theta_D)` is the output of an
    autoregressive NN inputting :math:`\mathbf{x}`.

    Example usage:

    >>> from pyro.nn import AutoRegressiveNN
    >>> input_dim = 10
    >>> count_bins = 8
    >>> base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))
    >>> hidden_dims = [input_dim * 10, input_dim * 10]
    >>> param_dims = [count_bins, count_bins, count_bins - 1, count_bins]
    >>> hypernet = AutoRegressiveNN(input_dim, hidden_dims, param_dims=param_dims)
    >>> transform = SplineAutoregressive(input_dim, hypernet, count_bins=count_bins)
    >>> pyro.module("my_transform", transform)  # doctest: +SKIP
    >>> flow_dist = dist.TransformedDistribution(base_dist, [transform])
    >>> flow_dist.sample()  # doctest: +SKIP

    :param input_dim: Dimension of the input vector. Despite operating element-wise,
        this is required so we know how many parameters to store.
    :type input_dim: int
    :param autoregressive_nn: an autoregressive neural network whose forward call
        returns tuple of the spline parameters
    :type autoregressive_nn: callable
    :param count_bins: The number of segments comprising the spline.
    :type count_bins: int
    :param bound: The quantity :math:`K` determining the bounding box,
        :math:`[-K,K]\times[-K,K]`, of the spline.
    :type bound: float
    :param order: One of ['linear', 'quadratic'] specifying the order of the spline.
    :type order: string

    References:

    Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios. Neural
    Spline Flows. NeurIPS 2019.

    Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie. Invertible Generative
    Modeling using Linear Rational Splines. AISTATS 2020.

    """

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    autoregressive = True

    def __init__(
        self, input_dim, autoregressive_nn, count_bins=8, bound=3.0, order="linear"
    ):
        super(SplineAutoregressive, self).__init__(cache_size=1)
        self.arn = autoregressive_nn
        self.spline = ConditionalSpline(
            autoregressive_nn, input_dim, count_bins, bound, order
        )

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from
        the base distribution (or the output of a previous transform)
        """
        spline = self.spline.condition(x)
        y = spline(x)
        self._cache_log_detJ = spline._cache_log_detJ
        return y

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x. Uses a previously cached inverse if available, otherwise
        performs the inversion afresh.
        """
        input_dim = y.size(-1)
        x = torch.zeros_like(y)

        # NOTE: Inversion is an expensive operation that scales in the dimension of the input
        for _ in range(input_dim):
            spline = self.spline.condition(x)
            x = spline._inverse(y)

        self._cache_log_detJ = spline._cache_log_detJ
        return x

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log Jacobian
        """
        x_old, y_old = self._cached_x_y
        if x is not x_old or y is not y_old:
            # This call to the parent class Transform will update the cache
            # as well as calling self._call and recalculating y and log_detJ
            self(x)

        return self._cache_log_detJ.sum(-1)







class ConditionalSplineAutoregressive(ConditionalTransformModule):
    r"""
    An implementation of the autoregressive layer with rational spline bijections of
    linear and quadratic order (Durkan et al., 2019; Dolatabadi et al., 2020) that
    conditions on an additional context variable. Rational splines are functions
    that are comprised of segments that are the ratio of two polynomials (see
    :class:`~pyro.distributions.transforms.Spline`).

    The autoregressive layer uses the transformation,

        :math:`y_d = g_{\theta_d}(x_d)\ \ \ d=1,2,\ldots,D`

    where :math:`\mathbf{x}=(x_1,x_2,\ldots,x_D)` are the inputs,
    :math:`\mathbf{y}=(y_1,y_2,\ldots,y_D)` are the outputs, :math:`g_{\theta_d}` is
    an elementwise rational monotonic spline with parameters :math:`\theta_d`, and
    :math:`\theta=(\theta_1,\theta_2,\ldots,\theta_D)` is the output of a
    conditional autoregressive NN inputting :math:`\mathbf{x}` and conditioning on
    the context variable :math:`\mathbf{z}`.

    Example usage:

    >>> from pyro.nn import ConditionalAutoRegressiveNN
    >>> input_dim = 10
    >>> count_bins = 8
    >>> context_dim = 5
    >>> batch_size = 3
    >>> base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))
    >>> hidden_dims = [input_dim * 10, input_dim * 10]
    >>> param_dims = [count_bins, count_bins, count_bins - 1, count_bins]
    >>> hypernet = ConditionalAutoRegressiveNN(input_dim, context_dim, hidden_dims,
    ... param_dims=param_dims)
    >>> transform = ConditionalSplineAutoregressive(input_dim, hypernet,
    ... count_bins=count_bins)
    >>> pyro.module("my_transform", transform)  # doctest: +SKIP
    >>> z = torch.rand(batch_size, context_dim)
    >>> flow_dist = dist.ConditionalTransformedDistribution(base_dist,
    ... [transform]).condition(z)
    >>> flow_dist.sample(sample_shape=torch.Size([batch_size]))  # doctest: +SKIP

    :param input_dim: Dimension of the input vector. Despite operating element-wise,
        this is required so we know how many parameters to store.
    :type input_dim: int
    :param autoregressive_nn: an autoregressive neural network whose forward call
        returns tuple of the spline parameters
    :type autoregressive_nn: callable
    :param count_bins: The number of segments comprising the spline.
    :type count_bins: int
    :param bound: The quantity :math:`K` determining the bounding box,
        :math:`[-K,K]\times[-K,K]`, of the spline.
    :type bound: float
    :param order: One of ['linear', 'quadratic'] specifying the order of the spline.
    :type order: string

    References:

    Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios. Neural
    Spline Flows. NeurIPS 2019.

    Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie. Invertible Generative
    Modeling using Linear Rational Splines. AISTATS 2020.

    """

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, input_dim, autoregressive_nn, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.nn = autoregressive_nn
        self.kwargs = kwargs

    def condition(self, context):
        """
        Conditions on a context variable, returning a non-conditional transform of
        of type :class:`~pyro.distributions.transforms.SplineAutoregressive`.
        """

        # Note that nn.condition doesn't copy the weights of the ConditionalAutoregressiveNN
        cond_nn = partial(self.nn, context=context)
        cond_nn.permutation = cond_nn.func.permutation
        cond_nn.get_permutation = cond_nn.func.get_permutation
        return SplineAutoregressive(self.input_dim, cond_nn, **self.kwargs)
    
    
    
    
    
class ConditionalSplineCoupling(ConditionalTransformModule):


    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, input_dim, split_dim, hypernet, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.split_dim = split_dim
        self.nn = hypernet
        self.kwargs = kwargs
        


    def condition(self, context):
        """
        Conditions on a context variable, returning a non-conditional transform of
        of type :class:`~pyro.distributions.transforms.SplineAutoregressive`.
        """

        # Note that nn.condition doesn't copy the weights of the ConditionalAutoregressiveNN
        cond_nn = partial(self.nn, context=context)
        #cond_nn.permutation = cond_nn.func.permutation
        #cond_nn.get_permutation = cond_nn.func.get_permutation
        return SplineCoupling(self.input_dim, self.split_dim, cond_nn, **self.kwargs)

    
    
    
    
    
    
    
    
    
    



class AffineCouplingTanH(TransformModule):
    r"""
    An implementation of the affine coupling layer of RealNVP (Dinh et al., 2017)
    that uses the bijective transform,

        :math:`\mathbf{y}_{1:d} = \mathbf{x}_{1:d}`
        :math:`\mathbf{y}_{(d+1):D} = \mu + \sigma\odot\mathbf{x}_{(d+1):D}`

    where :math:`\mathbf{x}` are the inputs, :math:`\mathbf{y}` are the outputs,
    e.g. :math:`\mathbf{x}_{1:d}` represents the first :math:`d` elements of the
    inputs, and :math:`\mu,\sigma` are shift and translation parameters calculated
    as the output of a function inputting only :math:`\mathbf{x}_{1:d}`.

    That is, the first :math:`d` components remain unchanged, and the subsequent
    :math:`D-d` are shifted and translated by a function of the previous components.

    Together with :class:`~pyro.distributions.TransformedDistribution` this provides
    a way to create richer variational approximations.

    Example usage:

    >>> from pyro.nn import DenseNN
    >>> input_dim = 10
    >>> split_dim = 6
    >>> base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))
    >>> param_dims = [input_dim-split_dim, input_dim-split_dim]
    >>> hypernet = DenseNN(split_dim, [10*input_dim], param_dims)
    >>> transform = AffineCoupling(split_dim, hypernet)
    >>> pyro.module("my_transform", transform)  # doctest: +SKIP
    >>> flow_dist = dist.TransformedDistribution(base_dist, [transform])
    >>> flow_dist.sample()  # doctest: +SKIP

    The inverse of the Bijector is required when, e.g., scoring the log density of a
    sample with :class:`~pyro.distributions.TransformedDistribution`. This
    implementation caches the inverse of the Bijector when its forward operation is
    called, e.g., when sampling from
    :class:`~pyro.distributions.TransformedDistribution`. However, if the cached
    value isn't available, either because it was overwritten during sampling a new
    value or an arbitary value is being scored, it will calculate it manually.

    This is an operation that scales as O(1), i.e. constant in the input dimension.
    So in general, it is cheap to sample *and* score (an arbitrary value) from
    :class:`~pyro.distributions.transforms.AffineCoupling`.

    :param split_dim: Zero-indexed dimension :math:`d` upon which to perform input/
        output split for transformation.
    :type split_dim: int
    :param hypernet: a neural network whose forward call returns a real-valued mean
        and logit-scale as a tuple. The input should have final dimension split_dim
        and the output final dimension input_dim-split_dim for each member of the
        tuple.
    :type hypernet: callable
    :param dim: the tensor dimension on which to split. This value must be negative
        and defines the event dim as `abs(dim)`.
    :type dim: int
    :param log_scale_min_clip: The minimum value for clipping the log(scale) from
        the autoregressive NN
    :type log_scale_min_clip: float
    :param log_scale_max_clip: The maximum value for clipping the log(scale) from
        the autoregressive NN
    :type log_scale_max_clip: float

    References:

    [1] Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation
    using Real NVP. ICLR 2017.

    """

    bijective = True

    def __init__(
        self,
        split_dim,
        hypernet,
        *,
        dim=-1,
        log_scale_min_clip=-5.0,
        log_scale_max_clip=3.0
    ):
        super().__init__(cache_size=1)
        if dim >= 0:
            raise ValueError("'dim' keyword argument must be negative")

        self.split_dim = split_dim
        self.nn = hypernet
        self.dim = dim
        self._cached_log_scale = None
        self.log_scale_min_clip = log_scale_min_clip
        self.log_scale_max_clip = log_scale_max_clip

    @constraints.dependent_property(is_discrete=False)
    def domain(self):
        return constraints.independent(constraints.real, -self.dim)

    @constraints.dependent_property(is_discrete=False)
    def codomain(self):
        return constraints.independent(constraints.real, -self.dim)

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from
        the base distribution (or the output of a previous transform)
        """
        x1, x2 = x.split(
            [self.split_dim, x.size(self.dim) - self.split_dim], dim=self.dim
        )

        # Now that we can split on an arbitrary dimension, we have do a bit of reshaping...
        mean, log_scale = self.nn(x1.reshape(x1.shape[: self.dim] + (-1,)))
        mean = mean.reshape(mean.shape[:-1] + x2.shape[self.dim :])
        log_scale = log_scale.reshape(log_scale.shape[:-1] + x2.shape[self.dim :])

        #log_scale = clamp_preserve_gradients(
        #    log_scale, self.log_scale_min_clip, self.log_scale_max_clip
        #)
        log_scale = -1.0*torch.tanh(log_scale)

        
        self._cached_log_scale = log_scale

        y1 = x1
        y2 = torch.exp(log_scale) * x2 + mean
        return torch.cat([y1, y2], dim=self.dim)

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x. Uses a previously cached inverse if available, otherwise
        performs the inversion afresh.
        """
        y1, y2 = y.split(
            [self.split_dim, y.size(self.dim) - self.split_dim], dim=self.dim
        )
        x1 = y1

        # Now that we can split on an arbitrary dimension, we have do a bit of reshaping...
        mean, log_scale = self.nn(x1.reshape(x1.shape[: self.dim] + (-1,)))
        mean = mean.reshape(mean.shape[:-1] + y2.shape[self.dim :])
        log_scale = log_scale.reshape(log_scale.shape[:-1] + y2.shape[self.dim :])

        #log_scale = clamp_preserve_gradients(
        #    log_scale, self.log_scale_min_clip, self.log_scale_max_clip
        #)
        log_scale = -1.0*torch.tanh(log_scale)
        
        
        self._cached_log_scale = log_scale

        x2 = (y2 - mean) * torch.exp(-log_scale)
        return torch.cat([x1, x2], dim=self.dim)

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log jacobian
        """
        x_old, y_old = self._cached_x_y
        if self._cached_log_scale is not None and x is x_old and y is y_old:
            log_scale = self._cached_log_scale
        else:
            x1, x2 = x.split(
                [self.split_dim, x.size(self.dim) - self.split_dim], dim=self.dim
            )
            _, log_scale = self.nn(x1.reshape(x1.shape[: self.dim] + (-1,)))
            log_scale = log_scale.reshape(log_scale.shape[:-1] + x2.shape[self.dim :])
            #log_scale = clamp_preserve_gradients(
            #    log_scale, self.log_scale_min_clip, self.log_scale_max_clip
            #)
            log_scale = -1.0*torch.tanh(log_scale)
        
        
        return _sum_rightmost(log_scale, self.event_dim)



class ConditionalAffineCouplingTanH(ConditionalTransformModule):
    r"""
    An implementation of the affine coupling layer of RealNVP (Dinh et al., 2017)
    that conditions on an additional context variable and uses the bijective
    transform,

        :math:`\mathbf{y}_{1:d} = \mathbf{x}_{1:d}`
        :math:`\mathbf{y}_{(d+1):D} = \mu + \sigma\odot\mathbf{x}_{(d+1):D}`

    where :math:`\mathbf{x}` are the inputs, :math:`\mathbf{y}` are the outputs,
    e.g. :math:`\mathbf{x}_{1:d}` represents the first :math:`d` elements of the
    inputs, and :math:`\mu,\sigma` are shift and translation parameters calculated
    as the output of a function input :math:`\mathbf{x}_{1:d}` and a context
    variable :math:`\mathbf{z}\in\mathbb{R}^M`.

    That is, the first :math:`d` components remain unchanged, and the subsequent
    :math:`D-d` are shifted and translated by a function of the previous components.

    Together with :class:`~pyro.distributions.ConditionalTransformedDistribution`
    this provides a way to create richer variational approximations.

    Example usage:

    >>> from pyro.nn import ConditionalDenseNN
    >>> input_dim = 10
    >>> split_dim = 6
    >>> context_dim = 4
    >>> batch_size = 3
    >>> base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))
    >>> param_dims = [input_dim-split_dim, input_dim-split_dim]
    >>> hypernet = ConditionalDenseNN(split_dim, context_dim, [10*input_dim],
    ... param_dims)
    >>> transform = ConditionalAffineCoupling(split_dim, hypernet)
    >>> pyro.module("my_transform", transform)  # doctest: +SKIP
    >>> z = torch.rand(batch_size, context_dim)
    >>> flow_dist = dist.ConditionalTransformedDistribution(base_dist,
    ... [transform]).condition(z)
    >>> flow_dist.sample(sample_shape=torch.Size([batch_size]))  # doctest: +SKIP

    The inverse of the Bijector is required when, e.g., scoring the log density of a
    sample with :class:`~pyro.distributions.ConditionalTransformedDistribution`.
    This implementation caches the inverse of the Bijector when its forward
    operation is called, e.g., when sampling from
    :class:`~pyro.distributions.ConditionalTransformedDistribution`. However, if the
    cached value isn't available, either because it was overwritten during sampling
    a new value or an arbitary value is being scored, it will calculate it manually.

    This is an operation that scales as O(1), i.e. constant in the input dimension.
    So in general, it is cheap to sample *and* score (an arbitrary value) from
    :class:`~pyro.distributions.transforms.ConditionalAffineCoupling`.

    :param split_dim: Zero-indexed dimension :math:`d` upon which to perform input/
        output split for transformation.
    :type split_dim: int
    :param hypernet: A neural network whose forward call returns a real-valued mean
        and logit-scale as a tuple. The input should have final dimension split_dim
        and the output final dimension input_dim-split_dim for each member of the
        tuple. The network also inputs a context variable as a keyword argument in
        order to condition the output upon it.
    :type hypernet: callable
    :param log_scale_min_clip: The minimum value for clipping the log(scale) from
        the NN
    :type log_scale_min_clip: float
    :param log_scale_max_clip: The maximum value for clipping the log(scale) from
        the NN
    :type log_scale_max_clip: float

    References:

    Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using
    Real NVP. ICLR 2017.

    """

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(self, split_dim, hypernet, **kwargs):
        super().__init__()
        self.split_dim = split_dim
        self.nn = hypernet
        self.kwargs = kwargs

    def condition(self, context):
        cond_nn = partial(self.nn, context=context)
        return AffineCouplingTanH(self.split_dim, cond_nn, **self.kwargs)