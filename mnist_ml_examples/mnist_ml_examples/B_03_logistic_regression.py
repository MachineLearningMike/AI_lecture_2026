"""Multiclass logistic regression and visualization of learned pixel weights."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from torchvision.datasets import MNIST

train = MNIST("./data", train=True, download=True)
test = MNIST("./data", train=False, download=True)
n_train, n_test = 12000, 3000
X_train = train.data[:n_train].numpy().reshape(n_train, -1).astype("float32") / 255
y_train = train.targets[:n_train].numpy()
X_test = test.data[:n_test].numpy().reshape(n_test, -1).astype("float32") / 255
y_test = test.targets[:n_test].numpy()

model = LogisticRegression(max_iter=250, solver="lbfgs")
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(f"Accuracy on {n_test} test images: {accuracy_score(y_test, pred):.3f}")
fig, ax = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Logistic regression confusion matrix")
fig.tight_layout(); fig.savefig("logistic_confusion_matrix.png", dpi=150); plt.show()

weights = model.coef_.reshape(10, 28, 28)
limit = np.abs(weights).max()
fig, axes = plt.subplots(2, 5, figsize=(11, 5))
for digit, ax in enumerate(axes.flat):
    ax.imshow(weights[digit], cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_title(f"Class {digit}"); ax.axis("off")
fig.suptitle("Learned pixel weights: positive vs. negative evidence")
fig.tight_layout(); fig.savefig("logistic_weight_maps.png", dpi=150); plt.show()
