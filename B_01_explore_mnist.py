"""Explore MNIST images, labels, and class balance."""
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import MNIST

ds = MNIST(root="./data", train=True, download=True)
images, labels = ds.data.numpy(), ds.targets.numpy()
rng = np.random.default_rng(0)
fig, axes = plt.subplots(3, 6, figsize=(10, 5))
for ax, i in zip(axes.flat, rng.choice(len(ds), 18, replace=False)):
    ax.imshow(images[i], cmap="gray")
    ax.set_title(f"Label: {labels[i]}")
    ax.axis("off")
fig.suptitle("MNIST: handwritten digit examples")
fig.tight_layout()
fig.savefig("mnist_examples.png", dpi=150)
plt.show()

counts = np.bincount(labels, minlength=10)
plt.figure(figsize=(8, 4))
plt.bar(np.arange(10), counts)
plt.xticks(np.arange(10))
plt.xlabel("Digit label")
plt.ylabel("Number of images")
plt.title("MNIST training-set class balance")
plt.tight_layout()
plt.savefig("mnist_class_balance.png", dpi=150)
plt.show()
