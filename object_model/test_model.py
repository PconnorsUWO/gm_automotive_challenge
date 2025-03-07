import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Load the label file with Polars
df = pd.read_csv(
    "labels_test_8_bit.txt",
    sep=" ",
    names=["image_file", "class_id", "x_min", "x_max", "y_min", "y_max"]
)


# Assume a dictionary mapping each image_file to its precomputed image feature tensor.
# Replace the dummy feature generation below with your actual image feature extraction/loading.
features_dict = {}
for img in df['image_file'].unique():
    # Example: using a random tensor to simulate precomputed features (feature vector dim = 256)
    features_dict[img] = torch.randn(256)

class DetectionDataset(Dataset):
    def __init__(self, df, features_dict):
        self.df = df
        self.features_dict = features_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_file = row['image_file']
        # Get the precomputed features for this image
        features = self.features_dict[image_file]
        # Bounding box: [x_min, x_max, y_min, y_max]
        bbox = torch.tensor([row['x_min'], row['x_max'], row['y_min'], row['y_max']], dtype=torch.float)
        # Adjust class_id (which is 1,2,3) to 0-indexed labels (0,1,2)
        label = torch.tensor(row['class_id'] - 1, dtype=torch.long)
        return features, bbox, label

# Create the dataset and dataloader
dataset = DetectionDataset(df, features_dict)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define a minimal baseline object detection model
class BaselineDetector(nn.Module):
    def __init__(self, input_dim=256):
        super(BaselineDetector, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        # Bounding box regression head: predicts 4 values (x_min, x_max, y_min, y_max)
        self.bbox_head = nn.Linear(128, 4)
        # Classification head: predicts logits for 3 classes (person, bicycle, vehicle)
        self.class_head = nn.Linear(128, 3)

    def forward(self, x):
        features = self.fc(x)
        bbox = self.bbox_head(features)
        class_logits = self.class_head(features)
        return bbox, class_logits

# Initialize model, loss functions, and optimizer
model = BaselineDetector(input_dim=256)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion_bbox = nn.MSELoss()         # Regression loss for bounding boxes
criterion_cls = nn.CrossEntropyLoss()   # Classification loss for target classes

num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0.0
    for features, bbox, label in dataloader:
        optimizer.zero_grad()
        pred_bbox, pred_class = model(features)
        loss_bbox = criterion_bbox(pred_bbox, bbox)
        loss_cls = criterion_cls(pred_class, label)
        loss = loss_bbox + loss_cls
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")
