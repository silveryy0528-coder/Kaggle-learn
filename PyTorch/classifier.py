#%%
'''
### PyTorch exercise ###

1.  Typical workflow
    Dataset -> split -> DataLoader -> training loop

2.  Steps in training loop
    Forward pass -> compute loss -> backward -> optimizer step

'''
import warnings
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader


# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
warnings.filterwarnings("ignore") # to clean up output cells
warnings.filterwarnings("ignore", message="Clipping input data to the valid range")

# Set hyper-parameters
train_test_split = 0.3
manual_seed = 42
batch_size = 16
show_images = True
learning_rate = 1e-3
num_epochs = 20

class Convnet(nn.Module):
    def __init__(self, conv_channels=[32, 64, 128], fc_channels=6):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=conv_channels[0],
            kernel_size=3,
            padding='same'
        )
        self.bn1 = nn.BatchNorm2d(num_features=conv_channels[0])

        self.conv2 = nn.Conv2d(
            in_channels=conv_channels[0],
            out_channels=conv_channels[1],
            kernel_size=3,
            padding='same'
        )
        self.bn2 = nn.BatchNorm2d(num_features=conv_channels[1])

        self.conv3 = nn.Conv2d(
            in_channels=conv_channels[1],
            out_channels=conv_channels[2],
            kernel_size=3,
            padding='same'
        )
        self.bn3 = nn.BatchNorm2d(num_features=conv_channels[2])

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # dummy tensor for feed forward layers
        dummy = torch.zeros(1, 3, 224, 224)
        dummy = self.pool(self.relu(self.conv1(dummy)))
        dummy = self.pool(self.relu(self.conv2(dummy)))
        dummy = self.pool(self.relu(self.conv3(dummy)))

        self.fc1 = nn.Linear(in_features=dummy.numel(), out_features=6)
        self.bn_fc1 = nn.BatchNorm1d(fc_channels)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(in_features=fc_channels, out_features=2)

    def forward(self, x):
        # BN before ReLU; it normalizes the output of conv layer so the ReLU sees
        # zero-centered data -> helps training converges faster
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))

        x = torch.flatten(x, 1)
        x = self.dropout(self.relu(self.bn_fc1(self.fc1(x))))
        x = self.fc2(x)

        return x


mean = [0.485,0.456,0.406]
std = [0.229,0.224,0.225]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

#%% Prepare data
data_folder = r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\PyTorch\data\animals'
dataset = datasets.ImageFolder(data_folder, transform=transform)

train_size = int((1 - train_test_split) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(
    dataset=dataset,
    lengths=[train_size, val_size],
    generator=torch.Generator().manual_seed(manual_seed)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)
print(f'{len(train_dataset)} training images belonging to {len(dataset.classes)} classes')
print(f'{len(val_dataset)} validation images belonging to {len(dataset.classes)} classes')

if show_images:
    class_names = dataset.classes
    images, labels = next(iter(train_loader))

    plt.figure(figsize=(10, 6))

    for i in range(8):
        img = (
            images[i] * torch.tensor(std).view(3, 1, 1)
            + torch.tensor(mean).view(3, 1, 1)
        )
        img = torch.clamp(img, 0, 1)

        plt.subplot(2, 4, i+1)
        plt.imshow(img.permute(1, 2, 0))
        plt.title(class_names[labels[i]])
        plt.axis("off")

    plt.tight_layout()
    plt.show()

#%% Training and validation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training will be done on {device}.')

model = Convnet(conv_channels=[32, 64, 128], fc_channels=6)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=2
)

train_losses = np.full(num_epochs, np.nan)
val_losses = np.full(num_epochs, np.nan)
train_accs = np.full(num_epochs, np.nan)
val_accs = np.full(num_epochs, np.nan)

best_val_loss = float('inf')
patience = 3
counter = 0

for epoch in range(num_epochs):
    # --- training loop ---
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # In PyTorch, gradients accumulate by default. To prevent this accumulation
        # and ensure that the optimizer updates the weights based only on the
        # gradients from the current batch, we must manually reset the
        # gradients before calculating them for the next iteration.
        optimizer.zero_grad()

        outputs = model(images)             # forward pass
        loss = criterion(outputs, labels)   # compute loss
        loss.backward()                     # backprop
        optimizer.step()                    # update weights

        # The returned loss is the loss averaged over the batch; hence, we need to
        # multiply it with the batch size to get the total contribution.
        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        running_corrects += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = running_corrects / total

    # --- validation loop ---
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    val_total = 0

    # Tells PyTorch not to track gradient to save memory and speeds up computation.
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            val_corrects += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss / val_total
    val_acc = val_corrects / val_total

    train_losses[epoch] = train_loss
    train_accs[epoch] = train_acc
    val_losses[epoch] = val_loss
    val_accs[epoch] = val_acc

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Learning rate: {optimizer.param_groups[0]['lr']}, "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # --- scheduler step (pass validation loss) ---
    scheduler.step(val_loss)

    # --- early stopping check ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            model.load_state_dict(torch.load("best_model.pth"))  # restore best
            break


#%% Plotting results
'''
Interpret loss values:
    1. Perfect prediction → loss ≈ 0
    2. Random guess for 2 classes → loss ≈ 0.69 (since -log(0.5) ≈ 0.693)
    3. Completely wrong prediction → loss can be >1

Loss is not always perfectly correlated to accuracy because it also captures
confidence through softmax probabilities.
'''
x = range(1, num_epochs + 1)
fig = plt.figure()
fig.add_subplot(2, 1, 1)
plt.plot(x, train_losses, label='training loss')
plt.plot(x, val_losses, label='val loss')
plt.xlabel('epochs')
plt.legend()
fig.add_subplot(2, 1, 2)
plt.plot(x, train_accs, label='training accuracy')
plt.plot(x, val_accs, label='val accuracy')
plt.xlabel('epochs')
plt.legend()
plt.show()