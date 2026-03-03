#%%
import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

sys.path.append(r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\PyTorch')
import utils


# Set hyperparameters and configurations
manual_seed = 42
train_val_split = 0.3
batch_size = 32
num_epochs = 30
show_images = True
learning_rate = 5e-6

mean, std = utils.retrieve_imagenet_mean_std()
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
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


#%% Prepare dataset and dataloaders
root_dir = r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\PyTorch\data\butterflies\train'
csv_file = r"C:\Users\guoya\Documents\Git_repo\Kaggle-learn\PyTorch\data\butterflies\Training_set.csv"

df = pd.read_csv(csv_file)

# Encode labels
class_names, class_to_idx = utils.retrieve_class_names(df)
num_classes = len(class_names)
df['label_idx'] = df['label'].map(class_to_idx)

train_df, val_df = train_test_split(
    df,
    test_size=train_val_split, 
    random_state=manual_seed,
    stratify=df['label_idx']    # very class keeps the same proportion
)

train_dataset = utils.ButterflyDataset(train_df, root_dir, transform=transform_train)
val_dataset = utils.ButterflyDataset(val_df, root_dir, transform=transform_val)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

if show_images:
    utils.visualize_samples(train_loader, class_names, mean, std)

#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = utils.retrieve_resnet_model(num_classes)
model.to(device)

weights_tensor = utils.retrieve_class_weights(train_df).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=weights_tensor)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=learning_rate
)
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

# Early stopping variables
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

        optimizer.zero_grad()

        outputs = model(images)             # forward pass
        loss = criterion(outputs, labels)   # compute loss
        loss.backward()                     # backprop
        optimizer.step()                    # update weights

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
        torch.save({
            "model_state_dict": model.state_dict(),
            "class_to_idx": class_to_idx,
        }, "best_model_val_split.pth")
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}, "
                  f"loading best model with val loss {best_val_loss:.4f}")
            model.load_state_dict(
                torch.load("best_model_val_split.pth")['model_state_dict'])
            break


#%%
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

#%%
cm = confusion_matrix(all_labels, all_preds)
cm_no_diag = cm.copy()
np.fill_diagonal(cm_no_diag, 0)

# Get top 10 confusion pairs
confused_pairs = np.dstack(np.unravel_index(
    np.argsort(cm_no_diag.ravel())[::-1],
    cm_no_diag.shape
))[0]

top_10 = confused_pairs[:10]
idx_to_class = {v:k for k,v in class_to_idx.items()}
for i, j in top_10:
    print(f"True: {idx_to_class[i]} → Predicted: {idx_to_class[j]}")


report = classification_report(all_labels, all_preds, target_names=class_names)
print("Classification Report:")
print(report)
