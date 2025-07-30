import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# PRimero hacer una barrido con un LR grande y luego ir ajustandolo, un ajuste grosero.
# ====== Parameters ======
TRAIN_SIZE = 10000   # Número de imágenes usadas para entrenamiento
EPOCHS = 50          # Cantidad de épocas (veces que se recorre el dataset de entrenamiento)
BATCH_SIZE = 32      # Tamaño del batch (número de imágenes procesadas juntas en cada paso)
LR = 0.5             # Tasa de aprendizaje (learning rate) del optimizador
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====== Data ======
transform = transforms.ToTensor()
train_full = datasets.MNIST(root='../1_mnist_plots/mnist_data', train=True, download=True, transform=transform)
train_set = Subset(train_full, range(TRAIN_SIZE))
test_set = datasets.MNIST(root='../1_mnist_plots/mnist_data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# ====== Model to be edited by students ======
class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128), # 28*28 tamaño de la imagen fijo, 128 es la cantidad de neuronas
            nn.ReLU(),
            nn.Linear(128, 10),
            #agrego otra capa
            nn.ReLU(),
            nn.Linear(10,10)
        )
    
    def forward(self, x):
        return self.net(x)

model = StudentNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR)

# ====== Training ======
print("Training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

# ====== Evaluation ======
print("\nEvaluating on test set...")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100.0 * correct / total
print(f"Final Accuracy on test set: {accuracy:.2f}%")

