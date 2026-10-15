"""Fully connected neural network with learning curves and predictions."""
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.ToTensor()
train_ds = Subset(datasets.MNIST("./data", train=True, download=True, transform=transform), range(12000))
test_ds = Subset(datasets.MNIST("./data", train=False, download=True, transform=transform), range(3000))
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256)
model = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Dropout(.2),
                      nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
losses, accuracies = [], []
for epoch in range(5):
    model.train(); total_loss = 0.0; seen = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); optimizer.step()
        total_loss += loss.item() * len(y); seen += len(y)
    losses.append(total_loss / seen)
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x.to(device)).argmax(1).cpu()
            correct += (pred == y).sum().item(); total += len(y)
    accuracies.append(correct / total)
    print(f"Epoch {epoch+1}: loss={losses[-1]:.4f}, test accuracy={accuracies[-1]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(range(1, 6), losses, marker="o"); axes[0].set(xlabel="Epoch", ylabel="Cross-entropy", title="Training loss")
axes[1].plot(range(1, 6), accuracies, marker="o"); axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Test accuracy")
fig.tight_layout(); fig.savefig("mlp_learning_curves.png", dpi=150); plt.show()

model.eval(); x, y = next(iter(test_loader))
with torch.no_grad(): probs = model(x.to(device)).softmax(1).cpu()
pred = probs.argmax(1).numpy()
fig, axes = plt.subplots(2, 5, figsize=(11, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(x[i, 0], cmap="gray")
    ax.set_title(f"T:{y[i].item()} P:{pred[i]} ({probs[i, pred[i]]:.2f})"); ax.axis("off")
fig.suptitle("MLP predictions: true / predicted / confidence")
fig.tight_layout(); fig.savefig("mlp_predictions.png", dpi=150); plt.show()
