#%%
import sys
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from torchvision import models
from torchvision import transforms
from torch.utils.data import DataLoader

sys.path.append(r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\PyTorch')
import utils

torch.backends.cudnn.benchmark = True

# Set hyperparameters and configurations
manual_seed = 42
batch_size = 32
num_epochs = 35
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

#%%
#%% Prepare dataset and dataloaders
root_dir = r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\PyTorch\data\butterflies\train'
csv_file = r"C:\Users\guoya\Documents\Git_repo\Kaggle-learn\PyTorch\data\butterflies\Training_set.csv"

df = pd.read_csv(csv_file)
class_names, class_to_idx = utils.retrieve_class_names(df)
num_classes = len(class_names)
df['label_idx'] = df['label'].map(class_to_idx)

full_dataset = utils.ButterflyDataset(
    dataframe=df,
    root_dir=root_dir,
    transform=transform_train
)
full_dataloader = DataLoader(full_dataset, batch_size=32, shuffle=True)

utils.visualize_samples(full_dataloader, class_names, mean, std)

#%% Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = utils.retrieve_resnet_model(num_classes)

model.to(device)

weights_tensor = utils.retrieve_class_weights(df).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=weights_tensor)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=learning_rate
)

train_losses = np.full(num_epochs, np.nan)
train_accs = np.full(num_epochs, np.nan)

for epoch in range(num_epochs):
    # --- training loop ---
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for images, labels in full_dataloader:
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

    train_losses[epoch] = train_loss
    train_accs[epoch] = train_acc

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

torch.save({
    "model_state_dict": model.state_dict(),
    "class_to_idx": class_to_idx,
}, "butterfly_final_model.pth")


x = range(1, num_epochs + 1)
fig = plt.figure()
fig.add_subplot(2, 1, 1)
plt.plot(x, train_losses, label='training loss')
plt.xlabel('epochs')
plt.legend()
fig.add_subplot(2, 1, 2)
plt.plot(x, train_accs, label='training accuracy')
plt.xlabel('epochs')
plt.legend()
plt.show()