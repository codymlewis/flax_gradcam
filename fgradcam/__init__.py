"""
A generic Grad-CAM library for the Flax neural network library. This enables the computation and visualization
of Grad-CAM for any convolutional layer in a neural network by adding 3 lines of code.
"""


from typing import Any, Optional

import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import jax
import flax.linen as nn
from flax.training.train_state import TrainState
import jax.numpy as jnp
import einops


def observe(module: nn.Module, x: jax.Array) -> jax.Array:
    """
    A flax module layer for Grad-CAM observation of the output that is passed in.
    Returns x with no changes, this simply adds model parameters for storing values for the Grad-CAM.

    Arguments:
    - module: Flax module to make the observation of
    - x: forward pass value to observe
    """
    x = module.perturb('gradcam_perturb', x)
    module.sow('intermediates', 'gradcam_sow', x)
    return x


def compute(train_state: TrainState, X: jax.Array) -> jax.Array:
    """
    Compute the Grad-CAM heatmap for model with respect to the input samples X.
    Returns a heatmap of the shape of the input to the observe layer but without channels,
    values are in the range [0, 1] where 0 shows least contribution and 1 shows most.

    Arguments:
    - model: Flax module that applies the neural network model
    - variables: Parameters and internal statistics of the model in a PyTree structure
    - X: Samples to compute the Grad-CAM of
    """
    _, state = train_state.apply_fn(train_state.params, X, mutable=["intermediates"])
    intermediates = state['intermediates']['gradcam_sow'][0]

    def _apply(params, perturbations, X):
        preds = train_state.apply_fn({'params': params, 'perturbations': perturbations}, X)
        return preds.max()

    params, perturbations = train_state.params['params'], train_state.params['perturbations']
    grads = jax.grad(_apply, argnums=1)(params, perturbations, X)
    grads = grads['gradcam_perturb']

    pooled_grads = einops.reduce(grads, 'b h w c -> b c', jnp.mean)
    conv_output = jnp.einsum('bhwc,bc->bhwc', intermediates, pooled_grads)
    heatmap = conv_output.mean(axis=-1)  # Average the channels of the heatmap
    heatmap = nn.relu(heatmap)
    # Finally, we normalize the heatmap into the [0, 1] range
    heatmap_maxs = einops.reduce(heatmap, 'b h w -> b', jnp.max)
    heatmap = (heatmap.T / heatmap_maxs).T
    return heatmap


def plot(
    x: jax.Array,
    heatmap: jax.Array,
    ax: Optional[plt.Axes] = None,
    channels_last: Optional[bool] = True,
    data_range: Optional[int] = 1,
    image_weight: Optional[float] = 0.5
):
    """
    Plot the interpolation between the heatmap and the x sample it was computed from. Note that this
    only operates on a single sample from the batch that the heatmap was computed from.

    Arguments:
    - x: A single sample for the respective heatmap
    - heatmap: A single heatmap that was computed via Grad-CAM
    - ax: Matplotlib axis to place the plot on
    - channels_last: Whether the image is in the channels-last format
    - data_range: Range for the values that x can have, assuming it starts from 0
    - image_weight: Amount of weight that the original image should have in the interpolation
    """
    # First we check whether the samples and heatmap are valid
    assert len(x.shape) < 4 and len(heatmap.shape) < 3, "This function can only handle single samples"
    p = channels_last and x.shape[-1] <= 3
    q = not channels_last and x.shape[0] <= 3
    assert p or q, "This function only handles a maximum of three channels"

    # scale the range of x into [0, 1]
    # if data_range appears incorrect find the nearest maximum bit value that fits the maximum value in the image
    if x.max() > data_range:
        data_range = 2**round(math.log(x.max(), 2))
    x = x / data_range

    # Next if x is single channel, then extend it to 3
    if len(x.shape) == 2:
        if channels_last:
            x = einops.repeat(x, "h w -> h w c", c=3)
        else:
            x = einops.repeat(x, "h w -> c h w", c=3)

    if (channels_last and x.shape[-1] < 3) or (not channels_last and x.shape[0] < 3):
        x = einops.repeat(x, "h w 1 -> h w 3")

    # Give the heatmap the jet colorscheme from matplotlib and delete the alpha channel
    heatmap = jnp.delete(mpl.colormaps['jet'](heatmap), 3, -1)
    if not channels_last:
        heatmap = einops.rearrange(heatmap, 'b h w c -> b c h w')

    # Make the final image to plot an interpolation between the sample and the heatmap
    x = (1 - image_weight) * heatmap + image_weight * x

    # Plot the image without an axis
    if ax is None:
        plt.imshow(x)
        plt.axis('off')
    else:
        ax.imshow(x)
        ax.axis('off')


def plot_heatmap(heatmap: jax.Array, ax: Optional[plt.Axes]):
    """
    Plot a single heatmap compute through Grad-CAM.

    Arguments:
    - heatmap: A single heatmap that was computed via Grad-CAM
    - ax: Matplotlib axis to place the plot on
    """
    # Check that the heatmap is valid
    assert len(heatmap.shape) < 3, "This function can only handle single heatmaps"

    # Give the heatmap the jet colorscheme and delete the alpha channel
    heatmap = jnp.delete(mpl.colormaps['jet'](heatmap), 3, -1)

    # Plot the image without an axis
    if ax is None:
        plt.imshow(heatmap)
        plt.axis('off')
    else:
        ax.imshow(heatmap)
        ax.axis('off')
