import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import random

# ====== Parameters ======
TRAIN_SIZE = 10000
EPOCHS = 10
BATCH_SIZE = 32
LR = 0.01
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====== Data ======
transform = transforms.ToTensor()
train_full = datasets.MNIST(root='../1_mnist_plots/mnist_data', train=True, download=True, transform=transform)
train_set = Subset(train_full, range(TRAIN_SIZE))
test_set = datasets.MNIST(root='../1_mnist_plots/mnist_data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# ====== LeNet-5 Model ======
class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # convolución con  kernels distintos
            # cuanto tengo batches el numero sera n en funcion de los batchess
            
            # podriamos probar cambiar el kernel_size o agregar otra etapa de convolucion
            
            nn.Conv2d(1, 6, kernel_size=10),             # 28 - 10 + 1 = 19 → 6x19x19
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),       # → 6x9x9
            # La cuenta me dice como ver la dimensionalidad que me queda de la imagen
            # Si cambio el kernel size deberia hacer nuevamente las cuentas
            nn.Conv2d(6, 16, kernel_size=5),             # 9 - 5 + 1 = 5 → 16x5x5
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),       # → 16x2x2
            # Hasta aca extraemos features
            nn.Flatten(),
            # 16 imagenes de 2x2, me da 64 componetes
            nn.Linear(16 * 2 * 2, 120),                  # 16×2×2 = 64
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.net(x)

model = StudentNet().to(device)
# Defino la perdida que usare las cross...
criterion = nn.CrossEntropyLoss()
# Optimizo con gradiente descendente
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
        outputs = model(images) # Lo paso por el modelo
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100.0 * correct / total
print(f"Final Accuracy on test set: {accuracy:.2f}%")

# ====== Visualization functions ======
def plot_kernels(tensor, title):
    n_kernels = tensor.shape[0]
    n_cols = 6
    n_rows = (n_kernels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten()

    for i in range(n_kernels):
        kernel = tensor[i, 0].detach().cpu()
        axes[i].imshow(kernel, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Kernel {i}')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_feature_maps(activations, title):
    n_maps = activations.shape[0]
    n_cols = 6
    n_rows = (n_maps + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten()

    for i in range(n_maps):
        axes[i].imshow(activations[i].cpu(), cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Map {i}')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# ====== Pick random test image and run forward pass ======
conv1 = model.net[0]  # Conv2d(1 → 6)
conv2 = model.net[3]  # Conv2d(6 → 16)

idx = random.randint(0, len(test_set) - 1)
image, label = test_set[idx]
input_image = image.unsqueeze(0).to(device)

with torch.no_grad():
    out1 = conv1(input_image)
    act1 = model.net[1](out1)
    act1_pool = model.net[2](act1)

    out2 = conv2(act1_pool)
    act2 = model.net[4](out2)
    act2_pool = model.net[5](act2)

# ====== Ordered Plots ======

# 1. Input image
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"Input Image - Label: {label}")
plt.axis('off')
plt.show()

# 2. Conv1 Kernels
plot_kernels(conv1.weight, "Conv1 Kernels (1→6)")

# 3. Feature Maps after Conv1 + Tanh
plot_feature_maps(act1.squeeze(), "Feature Maps after Conv1 + Tanh")

# 4. Conv2 Kernels
plot_kernels(conv2.weight[:, 0:1], "Conv2 Kernels (1st input channel, 6→16)")

# 5. Feature Maps after Conv2 + Tanh
plot_feature_maps(act2.squeeze(), "Feature Maps after Conv2 + Tanh")
