import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from zoobot.pytorch.training.finetune import FinetuneableZoobotClassifier

class DSPLTrainDataset(Dataset):
    def __init__(self, metadata_csv):
        # Filter metadata to keep ONLY rows meant for training
        all_df = pd.read_csv(metadata_csv)
        self.df = all_df[all_df['split'] == 'train'].reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        matrix = np.load(row['image_path'])
        m_min, m_max = matrix.min(), matrix.max()
        if (m_max - m_min) > 0:
            matrix = (matrix - m_min) / (m_max - m_min)
        tensor = torch.tensor(matrix).float().unsqueeze(0)
        return tensor, int(row['is_lens'])

print("Reading training data split...")
dataset = DSPLTrainDataset("dspl_test_metadata.csv")
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
print(f"Training subset loaded with {len(dataset)} items.")

model = FinetuneableZoobotClassifier(name='hf_hub:mwalmsley/zoobot-encoder-convnext_nano', num_classes=2, greyscale=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

print("\nActive Training Loop Starting...")
model.train()
for epoch in range(3):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    print(f"Epoch {epoch+1}/3 Completed | Average Training Loss: {running_loss / len(dataset):.4f}")

torch.save(model.state_dict(), "finetuned_dspl_lens_model.pt")
print("Successfully generated updated local weights file!")
