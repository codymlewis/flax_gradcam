from typing import Any, Optional

import matplotlib.pyplot as plt
import jax
import flax.linen as nn
import jax.numpy as jnp
import einops


PyTree = Any


def observe(module: nn.Module, x: jax.Array) -> jax.Array:
    x = module.perturb('gradcam_perturb', x)
    module.sow('intermediates', 'gradcam_sow', x)
    return x


def compute(model: nn.Module, variables: PyTree, X: jax.Array) -> jax.Array:
    _, state = model.apply(variables, X, mutable=["intermediates"])
    intermediates = state['intermediates']['gradcam_sow'][0]

    def _apply(params, perturbations, X):
        preds = model.apply({'params': params, 'perturbations': perturbations}, X)
        return preds.max()

    params, perturbations = variables['params'], variables['perturbations']
    grads = jax.grad(_apply, argnums=1)(params, perturbations, X)
    grads = grads['gradcam_perturb']

    pooled_grads = einops.reduce(grads, 'b h w c -> b c', jnp.mean)
    conv_output = jnp.einsum('bhwc,bc->bhwc', intermediates, pooled_grads)
    heatmap = conv_output.mean(axis=-1)
    heatmap_mins = einops.reduce(heatmap, 'b h w -> b', jnp.min)
    heatmap_maxs = einops.reduce(heatmap, 'b h w -> b', jnp.max)
    heatmap = ((heatmap.T - heatmap_mins) / (heatmap_maxs - heatmap_mins)).T
    return heatmap


def plot(
    x: jax.Array,
    heatmap: jax.Array,
    ax: Optional[plt.Axes] = None,
    channels_last: Optional[bool] = True,
    data_range: Optional[int] = 1,
):
    assert len(x.shape) < 4 and len(heatmap.shape) < 3, "This function can only handles single samples"
    # This function only handles a maximum of three channels
    assert (channels_last and x.shape[-1] <= 3) or (not channels_last and x.shape[0] <= 3)

    if len(x.shape) == 2:
        if channels_last:
            x = einops.repeat(x, "h w -> h w c", c=3)
        else:
            x = einops.repeat(x, "h w -> c h w", c=3)

    if (channels_last and x.shape[-1] < 3) or (not channels_last and x.shape[0] < 3):
        x = einops.repeat(x, "h w 1 -> h w 3")

    if channels_last:
        x[:, :, 0] = jnp.clip(x[:, :, 0] + heatmap, 0, data_range)
    else:
        x[0, :, :] = jnp.clip(x[0, :, :] + heatmap, 0, data_range)

    if ax is None:
        plt.imshow(x)
        plt.axis('off')
    else:
        ax.imshow(x)
        ax.axis('off')
