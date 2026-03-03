import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset
from PIL import Image


def retrieve_resnet_model(num_classes):

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    for param in model.layer4.parameters():
        param.requires_grad = True

    for param in model.layer3.parameters():
        param.requires_grad = True

    return model


def retrieve_class_weights(df):
    class_counts = df['label_idx'].value_counts().sort_index()
    weights = 1.0 / class_counts
    weights = weights / weights.sum()

    return torch.tensor(weights.values, dtype=torch.float)


def retrieve_imagenet_mean_std():
    # ImageNet dataset mean and std for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return mean, std


def retrieve_class_names(df):
    class_names = sorted(df['label'].unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

    return class_names, class_to_idx


def visualize_samples(train_loader, class_names, mean, std):
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


class ButterflyDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

        self.classes = sorted(self.data['label'].unique())
        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['filename']
        label = self.data.iloc[idx]['label_idx']

        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
