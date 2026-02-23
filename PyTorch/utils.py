import matplotlib.pyplot as plt
import torch


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