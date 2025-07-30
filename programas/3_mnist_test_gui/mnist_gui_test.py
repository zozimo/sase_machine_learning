import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import io
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

# ======== Hyperparameters ========
TRAIN_SIZE = 10000
EPOCHS = 10
BATCH_SIZE = 32
LR = 0.01
device = torch.device('cpu')

# ======== Define the MLP model ========
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleMLP().to(device)

# ======== Prepare MNIST dataset and train ========
print("Training on MNIST (10000 samples)...")
transform = transforms.ToTensor()
train_full = datasets.MNIST(root='../1_mnist_plots/mnist_data', train=True, download=True, transform=transform)
train_set = Subset(train_full, range(TRAIN_SIZE))
test_set = datasets.MNIST(root='../1_mnist_plots/mnist_data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
# Se toma un criterio y se optimiza
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR)

model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(train_loader):.4f}")

model.eval()
print("Model ready. Launching GUI...\n")

# ======== Preprocess canvas image ========
def preprocess_image(pil_img):
    img = pil_img.convert('L')  # grayscale
    img = ImageOps.invert(img)
    img = ImageOps.autocontrast(img)

    # Resize to 20x20
    img = img.resize((20, 20), Image.LANCZOS)

    # Paste into 28x28 canvas
    canvas = Image.new('L', (28, 28), 255)
    canvas.paste(img, (4, 4))

    # Convert to numpy and find center of mass
    np_img = np.array(canvas)
    cy, cx = scipy.ndimage.center_of_mass(255 - np_img)  # inverted: white bg
    cy, cx = int(cy), int(cx)
    
    # Compute shift to center it
    shift_y = 14 - cy
    shift_x = 14 - cx
    np_img = scipy.ndimage.shift(np_img, shift=(shift_y, shift_x), mode='constant', cval=255)

    # Normalize and convert to tensor
    tensor = transforms.ToTensor()(Image.fromarray(np_img.astype(np.uint8)))
    tensor = transforms.Normalize((0.5,), (0.5,))(tensor)
    return tensor.unsqueeze(0)

# ======== GUI App ========
class DigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Digit (0–9)")
        self.canvas_size = 200

        self.canvas = tk.Canvas(root, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.pack()

        btn_frame = tk.Frame(root)
        btn_frame.pack()
        tk.Button(btn_frame, text="Predict", command=self.predict).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Clear", command=self.clear).pack(side=tk.LEFT)

        self.image = Image.new("RGB", (self.canvas_size, self.canvas_size), "white")
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        r = 8
        x, y = event.x, event.y
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black', outline='black')
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill='black')

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_size, self.canvas_size], fill='white')

    def predict(self):
        img_tensor = preprocess_image(self.image)
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1).squeeze()  # shape: [10]

        digits = list(range(10))
        values = [p.item() for p in probs]

        # Filter out near-zero values
        non_zero_digits = [d for d, v in zip(digits, values) if v > 0.001]
        non_zero_values = [v for v in values if v > 0.001]

        # Find index of the max probability
        max_idx = non_zero_values.index(max(non_zero_values))

        # Set bar colors: red for max, blue for the rest
        colors = ['red' if i == max_idx else 'skyblue' for i in range(len(non_zero_values))]

        # Plot
        plt.figure(1, figsize=(6, 4))
        plt.clf()
        plt.bar(non_zero_digits, non_zero_values, color=colors)
        plt.title("Digit Probabilities")
        plt.xlabel("Digit")
        plt.ylabel("Confidence")
        plt.xticks(non_zero_digits)
        plt.ylim(0, 1.0)
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

# ======== Run the App ========
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitApp(root)
    root.mainloop()

