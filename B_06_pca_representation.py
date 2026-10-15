"""Unsupervised learning: PCA projects digit images into two dimensions."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from torchvision.datasets import MNIST

ds = MNIST("./data", train=True, download=True)
rng = np.random.default_rng(0); idx = rng.choice(len(ds), 5000, replace=False)
X = ds.data[idx].numpy().reshape(len(idx), -1).astype("float32") / 255
y = ds.targets[idx].numpy()
pca = PCA(n_components=2, random_state=0)
Z = pca.fit_transform(X)
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total explained by 2 PCs:", pca.explained_variance_ratio_.sum())
plt.figure(figsize=(9,7))
sc=plt.scatter(Z[:,0],Z[:,1],c=y,cmap="tab10",s=8,alpha=.65)
plt.xlabel("First principal component"); plt.ylabel("Second principal component")
plt.title("MNIST projected onto first two principal components")
plt.colorbar(sc,ticks=range(10),label="Digit label")
plt.tight_layout(); plt.savefig("pca_mnist_2d.png",dpi=150); plt.show()
fig,axes=plt.subplots(1,2,figsize=(7,3))
for i,ax in enumerate(axes): ax.imshow(pca.components_[i].reshape(28,28),cmap="coolwarm"); ax.set_title(f"PC {i+1}"); ax.axis("off")
fig.suptitle("Principal component directions"); fig.tight_layout(); fig.savefig("pca_components.png",dpi=150); plt.show()
