# Flax Grad-CAM

A generalized implementation of [Selvaraju et al.'s Grad-CAM](https://arxiv.org/abs/1610.02391) for the Flax neural network
library.

## How to use

This library works for any convolutional neural network written in [Flax](https://github.com/google/flax), that has image
input.

First install it with:

```sh
pip install -U git+https://github.com/codymlewis/flax_gradcam.git
```

Then make sure import `fgradcam` to use this library.

Finally, there are three lines of code needed to compute the Grad-CAM heatmap and plot it. The first line
is added to the Flax linen module, after the convolutional layer that you want to analyse:

```python
x = fgradcam.observe(self, x)
```

With that in place, after training the model, we compute the Grad-CAM heatmaps on the desired samples `X` with:

```python
heatmaps = fgradcam.compute(model, variables, X)
```

Finally, a heatmap can be visualized with:

```python
fgradcam.plot(X[0], heatmaps[0])
plt.show()  # Assuming matplotlib.pyplot was imported as plt
```


A full sample is shown in `samples/cnn.py`, in addition to an example of performing the Grad-CAM computation on a pretrained
model in `samples/transfer.ipynb`
