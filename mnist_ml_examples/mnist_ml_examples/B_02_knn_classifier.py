"""Classical ML: k-nearest-neighbors, confusion matrix, and mistakes."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from torchvision.datasets import MNIST

train = MNIST("./data", train=True, download=True)
test = MNIST("./data", train=False, download=True)
n_train, n_test = 8000, 2000  # Increase for a stronger, slower experiment.
X_train = train.data[:n_train].numpy().reshape(n_train, -1).astype("float32") / 255
y_train = train.targets[:n_train].numpy()
X_test = test.data[:n_test].numpy().reshape(n_test, -1).astype("float32") / 255
y_test = test.targets[:n_test].numpy()

model = KNeighborsClassifier(n_neighbors=3, weights="distance", n_jobs=-1)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(f"Accuracy on {n_test} test images: {accuracy_score(y_test, pred):.3f}")
fig, ax = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, cmap="Blues", colorbar=False)
ax.set_title("3-nearest-neighbors confusion matrix")
fig.tight_layout(); fig.savefig("knn_confusion_matrix.png", dpi=150); plt.show()

wrong = np.flatnonzero(pred != y_test)[:15]
fig, axes = plt.subplots(3, 5, figsize=(10, 6))
for ax, i in zip(axes.flat, wrong):
    ax.imshow(X_test[i].reshape(28, 28), cmap="gray")
    ax.set_title(f"True {y_test[i]} / pred {pred[i]}")
    ax.axis("off")
for ax in axes.flat[len(wrong):]: ax.axis("off")
fig.suptitle("KNN misclassified examples")
fig.tight_layout(); fig.savefig("knn_mistakes.png", dpi=150); plt.show()
