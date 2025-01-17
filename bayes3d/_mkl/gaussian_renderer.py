# AUTOGENERATED! DO NOT EDIT! File to edit: ../../scripts/_mkl/notebooks/06a - Gaussian Renderer.ipynb.

# %% auto 0
__all__ = ['normal_cdf', 'normal_pdf', 'normal_logpdf', 'inv', 'key', 'Array', 'Shape', 'Bool', 'Float', 'Int', 'FaceIndex',
           'FaceIndices', 'ArrayN', 'Array3', 'Array2', 'ArrayNx2', 'ArrayNx3', 'Matrix', 'PrecisionMatrix',
           'CovarianceMatrix', 'CholeskyMatrix', 'SquareMatrix', 'Vector', 'Direction', 'BaseVector', 'cast_rays',
           'ellipsoid_embedding', 'bilinear', 'log_gaussian', 'gaussian', 'gaussian_normalizing_constant',
           'gaussian_restriction_to_ray', 'discrete_arrival_probabilities', 'gaussian_time_of_arrival',
           'gaussian_most_likely_time_of_arrival', 'weighted_arrival_intersection', 'argmax_intersection',
           'weighted_argmax_intersection']

# %% ../../scripts/_mkl/notebooks/06a - Gaussian Renderer.ipynb 3
import bayes3d as b3d
import trimesh
import os
from bayes3d._mkl.utils import *
import matplotlib.pyplot as plt
import numpy as np
import jax
from jax import jit, vmap
import jax.numpy as jnp
from functools import partial
from bayes3d.camera import Intrinsics, K_from_intrinsics, camera_rays_from_intrinsics
from bayes3d.transforms_3d import transform_from_pos_target_up, add_homogenous_ones, unproject_depth
import tensorflow_probability as tfp
from tensorflow_probability.substrates.jax.math import lambertw

normal_cdf    = jax.scipy.stats.norm.cdf
normal_pdf    = jax.scipy.stats.norm.pdf
normal_logpdf = jax.scipy.stats.norm.logpdf
inv = jnp.linalg.inv

key = jax.random.PRNGKey(0)

# %% ../../scripts/_mkl/notebooks/06a - Gaussian Renderer.ipynb 5
from typing import Any, NamedTuple
import jaxlib


Array = np.ndarray | jax.Array
Shape = int | tuple[int, ...]
Bool = Array
Float = Array
Int = Array
FaceIndex = int
FaceIndices = Array
ArrayN      = Array
Array3      = Array
Array2      = Array
ArrayNx2    = Array
ArrayNx3    = Array
Matrix      = jaxlib.xla_extension.ArrayImpl
PrecisionMatrix  = Matrix
CovarianceMatrix = Matrix
CholeskyMatrix   = Matrix
SquareMatrix     = Matrix
Vector     = Array
Direction  = Vector
BaseVector = Vector

# %% ../../scripts/_mkl/notebooks/06a - Gaussian Renderer.ipynb 6
def ellipsoid_embedding(cov:CovarianceMatrix) -> Matrix:
    """Returns A with cov = A@A.T"""
    sigma, U = jnp.linalg.eigh(cov)
    D = jnp.diag(jnp.sqrt(sigma))
    return U @ D @ jnp.linalg.inv(U)

# %% ../../scripts/_mkl/notebooks/06a - Gaussian Renderer.ipynb 7
def bilinear(x:Array, y:Array, A:Matrix) -> Float:
    return x.T @ A @ y

# %% ../../scripts/_mkl/notebooks/06a - Gaussian Renderer.ipynb 8
def log_gaussian(x:Vector, mu:Vector, P:PrecisionMatrix) -> Float:
    """Evaluate an **unnormalized** gaussian at a given point."""
    return -0.5 * bilinear(x-mu, x-mu, P)


def gaussian(x:Vector, mu:Vector, P:PrecisionMatrix) -> Float:
    """Evaluate an **unnormalized** gaussian at a given point."""
    return jnp.exp(-0.5 * bilinear(x-mu, x-mu, P))


def gaussian_normalizing_constant(P:PrecisionMatrix) -> Float:
    """Returns the normalizing constant of an unnormalized gaussian."""
    n = P.shape[0]
    return jnp.sqrt(jnp.linalg.det(P)/(2*jnp.pi)**n)

# %% ../../scripts/_mkl/notebooks/06a - Gaussian Renderer.ipynb 9
def gaussian_restriction_to_ray(loc:Vector, P:PrecisionMatrix, A:CholeskyMatrix, x:Vector, v:Direction):
    """
    Restricts a gaussian to a ray and returns 
    the mean `mu` and standard deviation `std`, s.t. we have 
    $$
        P(x + t*v | loc, cov) = P(x + mu*v | loc, cov) * N(t | mu, std)
    $$
    """
    mu  = bilinear(loc - x, v, P)/bilinear(v, v, P)
    std = 1/jnp.linalg.norm(inv(A)@v)
    return mu, std

# %% ../../scripts/_mkl/notebooks/06a - Gaussian Renderer.ipynb 11
def discrete_arrival_probabilities(occupancy_probs:Vector):
    """
    Given an vector of `n` occupancy probabilities of neighbouring pixels, 
    it returns a vector of length `n+1` containing the probabilities of stopping 
    at a each pixel (while traversing them left to right) or not stopping at all.

    The return array is given by:
    $$
        q_i = p_i \cdot \prod_{j=0}^{i-1} (1 - p_j)
        
    $$
    for $i=0,...,n-1$, and
    $$
        q_n = \prod_{j=1}^{n-1} (1 - p_j) = 1 - \sum_{i=0}^{n-1} q_i.
    $$

    This is basically the discrete version of time of first arrival $X$ for an imhomogenous poisson processes
    with rate function $\sigma(t)$:
    $$
        X(T) = \sigma(T)*\exp(- \int_0^T \sigma(t) \ dt).
    $$
    """
    transmittances       = jnp.concatenate([jnp.array([1.0]), jnp.cumprod(1-occupancy_probs)])
    extended_occupancies = jnp.concatenate([occupancy_probs, jnp.array([1.0])])
    return extended_occupancies * transmittances

# %% ../../scripts/_mkl/notebooks/06a - Gaussian Renderer.ipynb 13
def gaussian_time_of_arrival(xs, mu, sig, w=1.0):
    """
    Time of first arrival for a **single** weighted 1-dimensional Gaussian, i.e. returns an array of
    with entries
    $$
        Y(T) = w*g(T | \mu, \sigma)*\exp(- \int_0^T w*g(t | \mu, \sigma) \ dt).
    $$
    """
    ys = w*normal_pdf(xs, loc=mu, scale=sig) * jnp.exp(
            - w*normal_cdf(xs, loc=mu, scale=sig) 
            + w*normal_cdf(0.0, loc=mu, scale=sig))
    return ys 


def gaussian_most_likely_time_of_arrival(mu, sig, w=1.):
    """
    Returns the most likely time of first arrival
    for a single weighted 1-dimensional Gaussian, i.e. the argmax of 
    $$
        Y(T) = w*g(T | \mu, \sigma)*\exp(- \int_0^T w*g(t | \mu, \sigma) \ dt).
    $$
    """
    # TODO: Check if this is correct, cf. my notes.
    Z = jnp.sqrt(lambertw(1/(2*jnp.pi) * w**2))
    return mu - Z*sig

# %% ../../scripts/_mkl/notebooks/06a - Gaussian Renderer.ipynb 16
def weighted_arrival_intersection(mu:Vector, P:PrecisionMatrix, A:CholeskyMatrix, w:Float, x:Vector, v:Direction):
    """
    Returns the "intersection" of a ray with a gaussian which we define as
    the mode of the gaussian restricted to the ray.
    """
    t0, sig0 = gaussian_restriction_to_ray(mu, P, A, x, v)
    w0 = w*gaussian(t0*v, mu, P)
    Z = w0/gaussian_normalizing_constant(P)
    t = gaussian_most_likely_time_of_arrival(t0, sig0, Z)
    return t, w0

# %% ../../scripts/_mkl/notebooks/06a - Gaussian Renderer.ipynb 17
def argmax_intersection(mu:Vector, P:PrecisionMatrix, x:Vector, v:Direction):
    """
    Returns the "intersection" of a ray with a gaussian which we define as
    the mode of the gaussian restricted to the ray.
    """
    t = bilinear(mu - x, v, P)/bilinear(v, v, P)
    return t


#|export
def weighted_argmax_intersection(mu:Vector, P:PrecisionMatrix, w:Float, x:Vector, v:Direction):
    """
    Returns the "intersection" of a ray with a gaussian which we define as
    the mode of the gaussian restricted to the ray.
    """
    t = bilinear(mu - x, v, P)/bilinear(v, v, P)
    return t, w*gaussian(x + t*v, mu, P)

# %% ../../scripts/_mkl/notebooks/06a - Gaussian Renderer.ipynb 24
def _cast_ray(v, mus, precisions, colors, weights, zmax=2.0, bg_color=jnp.array([1.,1.,1.,1.])):
    # TODO: Deal with negative intersections behind the camera
    # TODO: Maybe switch to log probs?

    # Compute fuzzy intersections `xs` with Gaussians and 
    # their function values `sigmas`
    ts, sigmas = vmap(weighted_argmax_intersection, (0,0,0,None,None))(
                        mus, precisions, weights, jnp.zeros(3), v)
    order  = jnp.argsort(ts)
    ts     = ts[order]
    sigmas = sigmas[order]
    xs     = ts[:,None]*v[None,:]

    # TODO: Ensure that alphas are in [0,1]
    # TODO: Should we reset the color opacity to `op`?
    # Alternatively we can set `alphas = (1 - jnp.exp(-sigmas*1.0))` -- cf. Fuzzy Metaballs paper
    alphas = sigmas * (ts > 0)
    arrival_probs = discrete_arrival_probabilities(alphas)
    op = 1 - arrival_probs[-1] # Opacity
    mean_depth = jnp.sum(arrival_probs[:-1]*xs[:,2]) \
                    + arrival_probs[-1]*zmax
    mean_color = jnp.sum(arrival_probs[:-1,None]*colors[order], axis=0) \
                    + arrival_probs[-1]*bg_color    

    return mean_depth, mean_color, op


cast_rays = jit(vmap(_cast_ray, (0,None,None,None,None,None,None)))
