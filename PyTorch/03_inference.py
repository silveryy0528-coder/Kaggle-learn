#%%
import sys
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from torchvision import transforms

cwd = r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\PyTorch'
sys.path.append(cwd)
import utils

torch.backends.cudnn.benchmark = True

mean, std = utils.retrieve_imagenet_mean_std()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


def predict_image(image_path, model, idx_to_class, transform, device):
    model.eval()

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()

    predicted_class = idx_to_class[pred_idx]
    confidence = probs[0][pred_idx].item()
    
    return predicted_class, confidence


def retrieve_model(model_path):
    checkpoint = torch.load(model_path, map_location=device)

    model = utils.retrieve_resnet_model(num_classes=len(checkpoint['class_to_idx']))
    model.load_state_dict(checkpoint['model_state_dict'])

    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    return model, idx_to_class

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#%%
# Load the trained model
final_model_path = os.path.join(cwd, "butterfly_final_model.pth")
final_model, idx_to_class = retrieve_model(final_model_path)
final_model.to(device)

baseline_model_path = os.path.join(cwd, "best_model_val_split.pth")
baseline_model, _ = retrieve_model(baseline_model_path)
baseline_model.to(device)

test_folder = r'C:\Users\guoya\Documents\Git_repo\Kaggle-learn\PyTorch\data\butterflies\test'

results = []
for idx, filename in enumerate(os.listdir(test_folder)):
    if not filename.endswith('.jpg'):
        continue

    image_path = os.path.join(test_folder, filename)

    pred_final, conf_final = predict_image(
        image_path, final_model, idx_to_class, transform, device)
    pred_baseline, conf_baseline = predict_image(
        image_path, baseline_model, idx_to_class, transform, device)

    print(
        f"Image: {filename}, "
        f"Final Predicted: {pred_final}, "
        f"Final Confidence: {conf_final:.4f}, "
        f"Baseline Predicted: {pred_baseline}, "
        f"Baseline Confidence: {conf_baseline:.4f}")

    results.append({
        "filename": filename,
        "final_label": pred_final,
        "final_confidence": conf_final,
        "baseline_label": pred_baseline,
        "baseline_confidence": conf_baseline
    })

results_df = pd.DataFrame(results)

#%% Compare predictions between baseline and final models
diff_df = results_df[results_df['final_label'] != results_df['baseline_label']]

'''
Case A - Both low confidence
         Hard samples where both models struggle
Case B - Final model more confident
         Extra data improves represetation
Case C - baseline model more confident
         Validation split acted as a regularizer to prevent overfitting
'''

mean_conf_final = diff_df["final_confidence"].mean()
mean_conf_baseline = diff_df["baseline_confidence"].mean()
print(f"Mean confidence for final model on differing predictions: \
      {mean_conf_final:.4f}")
print(f"Mean confidence for baseline model on differing predictions: \
      {mean_conf_baseline:.4f}")

top5 = diff_df.sort_values(by="final_confidence", ascending=False).head()

for idx, row in top5.iterrows():
    image_path = os.path.join(test_folder, row['filename'])
    image = Image.open(image_path).convert('RGB')
    plt.imshow(image)
    plt.title(
        f"Final: {row['final_label']} ({row['final_confidence']:.4f})\n"
        f"Baseline: {row['baseline_label']} ({row['baseline_confidence']:.4f})"
    )
    plt.axis('off')
    plt.show()