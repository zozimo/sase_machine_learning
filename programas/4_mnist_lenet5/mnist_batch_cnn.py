import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

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

# ====== CNN Model: LeNet-5 ======
class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),  # output: 6x28x28
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),                # output: 6x14x14

            nn.Conv2d(6, 16, kernel_size=5),                      # output: 16x10x10
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),                # output: 16x5x5

            nn.Flatten(),                                         # 16*5*5 = 400
            nn.Linear(400, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10)
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
