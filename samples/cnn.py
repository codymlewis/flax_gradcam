import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import jaxopt
import datasets
import einops
from sklearn import metrics
from tqdm import trange
import fgradcam


def ce_loss(model):
    "A simple cross-entropy loss function"
    @jax.jit
    def _loss(params, X, Y):
        logits = jnp.clip(model.apply(params, X), 1e-15, 1 - 1e-15)
        one_hot = jax.nn.one_hot(Y, logits.shape[-1])
        return -jnp.mean(jnp.einsum("bl,bl -> b", one_hot, jnp.log(logits)))
    return _loss


def accuracy(model, variables, X, Y, batch_size=1000):
    """
    Calculate the accuracy of the model across the given dataset

    Arguments:
    - model: Model function that performs predictions given parameters and samples
    - variables: Parameters and other learned values used by the model
    - X: The samples
    - Y: The corresponding labels for the samples
    - batch_size: Amount of samples to compute the accuracy on at a time
    """
    @jax.jit
    def _apply(batch_X):
        return jnp.argmax(model.apply(variables, batch_X), axis=-1)
    preds, Ys = [], []
    for i in range(0, len(Y), batch_size):
        i_end = min(i + batch_size, len(Y))
        preds.append(_apply(X[i:i_end]))
        Ys.append(Y[i:i_end])
    return metrics.accuracy_score(jnp.concatenate(Ys), jnp.concatenate(preds))


def load_mnist():
    """
    Load the Fashion MNIST dataset http://arxiv.org/abs/1708.07747

    Arguments:
    - seed: seed value for the rng used in the dataset
    """
    ds = datasets.load_dataset("fashion_mnist")
    ds = ds.map(
        lambda e: {
            'X': einops.rearrange(np.array(e['image'], dtype=np.float32) / 255, "h (w c) -> h w c", c=1),
            'Y': e['label']
        },
        remove_columns=['image', 'label']
    )
    features = ds['train'].features
    features['X'] = datasets.Array3D(shape=(28, 28, 1), dtype='float32')
    ds['train'] = ds['train'].cast(features)
    ds['test'] = ds['test'].cast(features)
    ds.set_format('numpy')
    return ds


class CNN(nn.Module):
    "A simple CNN model"
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=48, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=16, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = fgradcam.observe(self, x)  # This is to compute the Grad-CAM later on
        x = einops.rearrange(x, "b h w c -> b (h w c)")
        x = nn.Dense(10)(x)
        x = nn.softmax(x)
        return x


if __name__ == "__main__":
    # Initialize the model and dataset
    dataset = load_mnist()
    X, Y = dataset['train']['X'], dataset['train']['Y']
    model = CNN()
    variables = model.init(jax.random.PRNGKey(42), X[:1])

    # Train the model
    solver = jaxopt.OptaxSolver(ce_loss(model), optax.sgd(0.1), maxiter=3000)
    state = solver.init_state(variables)
    step = jax.jit(solver.update)
    rng = np.random.default_rng()
    for i in (pbar := trange(solver.maxiter)):
        idx = rng.choice(len(Y), size=128, replace=False)
        params, state = step(params=variables, state=state, X=X[idx], Y=Y[idx])
        pbar.set_postfix_str(f"LOSS: {state.value:.3f}")
    final_acc = accuracy(model, params, dataset['test']['X'], dataset['test']['Y'])
    print(f"Final accuracy: {final_acc:.3%}")

    # Compute and plot the Grad-CAM
    batch_size = 25
    print("Computing Grad-CAM heatmap...")
    heatmap = fgradcam.compute(model, params, X[:batch_size])
    print("Done. Plotting the results...")
    fig, axes = plt.subplots(nrows=round(batch_size**0.5), ncols=round(batch_size**0.5))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        fgradcam.plot(X[i], heatmap[i], ax=ax)
    plt.tight_layout()
    plt.show()
