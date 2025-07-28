import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import json
from sklearn.metrics import classification_report

# Veriyi indirme ve hazırlama
transform = transforms.ToTensor()
mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
mnist_1k, _ = random_split(mnist, [1000, len(mnist) - 1000])
train_loader = DataLoader(mnist_1k, batch_size=32, shuffle=True)

# Model tanımı
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)

# Loss ve optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Eğitim döngüsü
losses = []
for epoch in range(5):
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")

# Kayıp grafiği çizme
plt.plot(losses, marker='o')
plt.title("Eğitim Kaybı")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("mlp_loss.png")

# Doğruluk metrikleri
all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in train_loader:
        x = x.to(device)
        preds = model(x)
        preds = preds.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())

report = classification_report(all_labels, all_preds, output_dict=True)
with open("mlp_metrics.json", "w") as f:
    json.dump(report, f, indent=4)

print("Eğitim tamamlandı. mlp_loss.png ve mlp_metrics.json oluşturuldu.")