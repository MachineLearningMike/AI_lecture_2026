"""CNN: learn spatial features; visualize curves, errors, and filters."""
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import ConfusionMatrixDisplay

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tf = transforms.ToTensor()
train_ds = Subset(datasets.MNIST("./data", train=True, download=True, transform=tf), range(15000))
test_ds = Subset(datasets.MNIST("./data", train=False, download=True, transform=tf), range(3000))
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256)
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                      nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32*7*7, 128), nn.ReLU(),
                                        nn.Dropout(.2), nn.Linear(128, 10))
    def forward(self, x): return self.classifier(self.features(x))
model = SmallCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss(); losses=[]; accuracies=[]
for epoch in range(5):
    model.train(); total_loss=0.; seen=0
    for x,y in train_loader:
        x,y=x.to(device),y.to(device); optimizer.zero_grad()
        loss=loss_fn(model(x),y); loss.backward(); optimizer.step()
        total_loss += loss.item()*len(y); seen += len(y)
    losses.append(total_loss/seen)
    model.eval(); correct=total=0
    with torch.no_grad():
        for x,y in test_loader:
            correct += (model(x.to(device)).argmax(1).cpu()==y).sum().item(); total += len(y)
    accuracies.append(correct/total)
    print(f"Epoch {epoch+1}: loss={losses[-1]:.4f}, test accuracy={accuracies[-1]:.3f}")

fig, axes=plt.subplots(1,2,figsize=(10,4))
axes[0].plot(range(1,6),losses,marker="o"); axes[0].set(xlabel="Epoch",ylabel="Cross-entropy",title="CNN training loss")
axes[1].plot(range(1,6),accuracies,marker="o"); axes[1].set(xlabel="Epoch",ylabel="Accuracy",title="CNN test accuracy")
fig.tight_layout(); fig.savefig("cnn_learning_curves.png",dpi=150); plt.show()
ys=[]; ps=[]; model.eval()
with torch.no_grad():
    for x,y in test_loader:
        ps.extend(model(x.to(device)).argmax(1).cpu().numpy()); ys.extend(y.numpy())
fig,ax=plt.subplots(figsize=(8,7)); ConfusionMatrixDisplay.from_predictions(ys,ps,ax=ax,cmap="Blues",colorbar=False)
ax.set_title("CNN confusion matrix"); fig.tight_layout(); fig.savefig("cnn_confusion_matrix.png",dpi=150); plt.show()
filters=model.features[0].weight.detach().cpu().numpy()
fig,axes=plt.subplots(2,8,figsize=(10,3))
for i,ax in enumerate(axes.flat): ax.imshow(filters[i,0],cmap="coolwarm"); ax.set_title(f"Filter {i+1}"); ax.axis("off")
fig.suptitle("Learned first-layer convolution filters"); fig.tight_layout(); fig.savefig("cnn_filters.png",dpi=150); plt.show()
