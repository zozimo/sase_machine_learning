import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import random

# ====== Use CPU (works well on Raspberry Pi) ======
device = torch.device('cpu')

# ====== Parameters ======
TRAIN_SIZE = 1000
EPOCHS = 50 # Number of epochs to train the model, 1000/32ls
# conviene promediar cada 32 imágenes porque evita caer en un mínimo local por el ruido del gradiente descente
BATCH_SIZE = 32 #Tengo un promedio cada 32 imágenes y va actualizando los pesos. 
LR = 0.01

# ====== Load datasets ======
transform = transforms.ToTensor()
full_train = datasets.MNIST(root='./mnist_data', train=True, download=True, transform=transform)
train_set = Subset(full_train, range(TRAIN_SIZE))
test_set = datasets.MNIST(root='./mnist_data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

# ====== Simple model ======
# 128 neuronas que van a tener distinta activación cada una
# La normalización no siempre esta, depende de la función de activación
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential( #defino la red
            nn.Flatten(),
            nn.Linear(28 * 28, 128),  #Tener cuidado que coincida la entrada salida, las dimensiones
            nn.ReLU(),
            nn.Linear(128, 10) # el 10 es un vector de 10 dimensiones
        )
        # Corro la función
    def forward(self, x): 
        return self.fc(x)

model = MLP().to(device) # Lo cargo al modelo en CPU o GPU
criterion = nn.CrossEntropyLoss() # Elijo el criterio de entropia
optimizer = optim.SGD(model.parameters(), lr=LR) # Uso el gradiendt Descente y obtengo los parametros de mi modelo

# ====== Train ======
print("Training model...\n")
# loop de entrenamiento
losses = []  # Guardar el loss promedio de cada epoch

for epoch in range(EPOCHS): # Esto corre el numero de EPOCHS
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device) # CArgo los datos al CPU o GPU
        pred = model(x) # Objeto de las capas cargado en RAM
        loss = criterion(pred, y) # 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # Actualiza los parametros
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

# Graficar la evolución del loss
plt.figure()
plt.plot(range(1, EPOCHS+1), losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss per Epoch')
plt.grid(True)
plt.show()

# ====== Show random 10 test images ======
print("\nShowing 10 random predictions from test set...")

# Select 10 random indices from the test set
indices = random.sample(range(len(test_set)), 10)

# Clear any existing figure
plt.close('all')

fig, axs = plt.subplots(2, 5, figsize=(10, 4))

model.eval()
with torch.no_grad(): # corre sin actualizar los parametros, ya lo entrene.
    for i, idx in enumerate(indices):
        x, y = test_set[idx]
        output = model(x.unsqueeze(0).to(device)) # Verctor de 10 componentes
        pred = output.argmax().item() # argmax me da el mayor para hacer mi predicción, no quiere decir que el resto sea cero

        ax = axs[i // 5, i % 5]
        ax.imshow(x.squeeze(), cmap='gray')
        ax.set_title(f"True: {y} / Pred: {pred}")
        ax.axis('off')

plt.tight_layout()
plt.show()
