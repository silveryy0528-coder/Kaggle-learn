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
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


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
num_epochs = 30

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

    def forward(self, x, return_features=False):
        # BN before ReLU; it normalizes the output of conv layer so the ReLU sees
        # zero-centered data -> helps training converges faster
        x1 = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x1)

        x2 = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x2)

        x3 = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x3)

        x = torch.flatten(x, 1)
        x = self.dropout(self.relu(self.bn_fc1(self.fc1(x))))
        out = self.fc2(x)

        if return_features:
            return out, x1, x2, x3

        return out


mean = [0.485,0.456,0.406]
std = [0.229,0.224,0.225]

# Online data augmentation to increase data diversity and prevent overfitting.
# The transformations are applied randomly during training, so the model sees a different
# version of the same image in each epoch.
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)), # prevents over-reliance on object position
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

#%% Prepare data
data_folder = r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\PyTorch\data\animals'

# ImageFolder only stores file paths and labels, and applies transformations on-the-fly
# when loading images. This is memory efficient and allows for data augmentation.
train_dataset = datasets.ImageFolder(data_folder, transform=transform_train)
val_dataset = datasets.ImageFolder(data_folder, transform=transform_val)
class_names = train_dataset.classes

train_size = int((1 - train_test_split) * len(train_dataset))
indices = torch.randperm(len(train_dataset))
train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

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
print(f'{len(train_dataset)} training images belonging to {len(class_names)} classes')
print(f'{len(val_dataset)} validation images belonging to {len(class_names)} classes')

if show_images:
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
    model.eval() # turn off dropout & use running batchnorm stats
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
            print(f"Early stopping at epoch {epoch+1}, "
                  f"loading best model with val loss {best_val_loss:.4f}")
            model.load_state_dict(torch.load("best_model.pth"))
            break


#%% Plotting results
'''
Visualise training and validation loss/accuracy curves to check for over-/underfitting.
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

#%%
'''
Visualise learned features to check if the model is learning meaningful representations.
Hireacrhical feature learning in CNNs:
Edges -> Corners / Textures -> Object parts -> Objects

Layer 1:
    Edge detectors, color contrast, simple shapes
Layer 2:
    More complex patterns, textures, corners, combinations of edges
Layer 3:
    Large regions activated, parts of objects
    (three 3x3 kernels can reach a receptive field of 7x7)
'''
if show_images:
    model.eval()

    img_idx = 5
    map_idx = 3

    images, labels = next(iter(val_loader))
    img = images[img_idx].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img, return_features=True)

    feature_maps = outputs[map_idx].cpu().squeeze(0)
    num_maps = feature_maps.shape[0]

    img = images[img_idx].cpu().numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)

    plt.figure(figsize=(12, 8))
    plt.subplot(4, 4, 1)
    plt.imshow(img)
    plt.axis('off')

    for i in range(min(num_maps, 15)):
        plt.subplot(4, 4, i+2)
        plt.imshow(feature_maps[i], cmap='viridis')
        plt.axis('off')

    plt.suptitle(f'Feature maps of image {img_idx} from convolutional layer {map_idx}')
    plt.tight_layout()
    plt.show()

#%% Confusion matrix and classification report
model.eval()

all_preds = []
all_labels = []
all_images = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_images.append(images.cpu())

all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)
all_images = torch.cat(all_images)

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

report = classification_report(all_labels, all_preds, target_names=class_names)
print("Classification Report:")
print(report)

misclassified_indices = torch.where(all_preds != all_labels)[0]
if show_images and len(misclassified_indices) > 0:
    plt.figure(figsize=(8, 4))

    for i, idx in enumerate(misclassified_indices):
        if i >= 4:
            break
        img = all_images[idx].numpy().transpose((1, 2, 0))
        img = std * img + mean
        img = np.clip(img, 0, 1)

        plt.subplot(1, 4, i+1)
        plt.imshow(img)
        plt.title(f"True: {class_names[all_labels[idx]]}\nPred: {class_names[all_preds[idx]]}")
        plt.axis('off')

    plt.suptitle('Misclassified Images')
    plt.tight_layout()
    plt.show()