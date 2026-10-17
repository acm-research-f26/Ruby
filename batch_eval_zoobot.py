import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# --- NEW: IMPORT ZOOBOT'S NATIVE CLASSIFIER UTILITIES ---
from zoobot.pytorch.training.finetune import FinetuneableZoobotClassifier

# Look inside your batch_eval_zoobot.py file around line 11 and swap out DSPLDataset with this:
class DSPLDataset(Dataset):
    """Loads and normalizes generated .npy simulation matrices into active memory."""
    def __init__(self, metadata_csv):
        # FIXED: ONLY pull rows marked for testing to guarantee a completely fair evaluation
        all_df = pd.read_csv(metadata_csv)
        self.df = all_df[all_df['split'] == 'test'].reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        matrix = np.load(row['image_path'])
        
        # Max-min normalization layerS
        m_min, m_max = matrix.min(), matrix.max()
        if (m_max - m_min) > 0:
            matrix = (matrix - m_min) / (m_max - m_min)
            
        # Convert to tensor format (Channels=1, Height=64, Width=64)
        tensor = torch.tensor(matrix).float().unsqueeze(0)
        
        meta_metrics = {
            'true_label': int(row['is_lens']),
            'magnitude': float(row['magnitude']),
            'arc_size': float(row['arc_size']),
            'blended': int(row['blended'])
        }
        return tensor, meta_metrics

print("Loading Zoobot Pretrained Checkpoint Engine...")
model = FinetuneableZoobotClassifier(
    name='hf_hub:mwalmsley/zoobot-encoder-convnext_nano', 
    num_classes=2,
    greyscale=True 
)

# === THE CRITICAL FIXED LINE: Load your fine-tuned local weights file ===
checkpoint_path = "finetuned_dspl_lens_model.pt"
model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model engine loaded successfully onto computing hardware: {device}")
print("Assembling batch pipeline for 200 generated balanced items...")
dataset = DSPLDataset("dspl_test_metadata.csv")
dataloader = DataLoader(dataset, batch_size=200, shuffle=False) 

for images_batch, metadata_batch in dataloader:
    print(f"Successfully stacked matrix block into tensor shape: {images_batch.shape}")
    images_batch = images_batch.to(device)
    
    with torch.no_grad():
        logits = model(images_batch)
        
        # Apply Softmax instead of Sigmoid since the model now outputs 2 distinct class nodes
        probabilities_tensor = torch.softmax(logits, dim=1)
        
        # Extract the probability specifically for the "Lens" class (Node Index 1)
        probabilities = probabilities_tensor[:, 1].cpu().numpy()

    # The prediction is simply whichever node has the higher probability value
    predictions = torch.argmax(probabilities_tensor, dim=1).cpu().numpy()

# === FIXED: Filter the analysis dataframe to match the 163 test items exactly ===
full_df = pd.read_csv("dspl_test_metadata.csv")
analysis_df = full_df[full_df['split'] == 'test'].reset_index(drop=True)

analysis_df['true_label'] = [t.item() for t in metadata_batch['true_label']]
analysis_df['predicted_label'] = predictions
analysis_df['pred_probability'] = probabilities

# 2. RENDER COMPREHENSIVE STATISTICAL BREAKDOWN SUMMARY
print(f"\n================ TOTAL OVERALL {len(analysis_df)} IMAGE PERFORMANCE ================")
print(classification_report(analysis_df['true_label'], analysis_df['predicted_label'], labels=[0, 1], target_names=['Non-Lens', 'Lens'], zero_division=0))

def compute_subset_metrics(df, subset_title, query_string):
    subset = df.query(query_string)
    if len(subset) == 0:
        print(f"{subset_title:<30} | Items: 0     | Metric: N/A")
        return
    accuracy = accuracy_score(subset['true_label'], subset['predicted_label'])
    print(f"{subset_title:<30} | Items: {len(subset):<5} | Accuracy Target: {accuracy:.3f}")

print("================ CONDITION ANALYSIS RESULTS ================")
# A. Faintness Binning Breakdown (Filters target rows where true lenses are active)
lenses_only = analysis_df.query("is_lens == 1").copy()
lenses_only['faint_bin'] = pd.cut(lenses_only['magnitude'], bins=[0, 13.5, 14.5, 30], labels=['Bright Sky Lenses', 'Medium Sky Lenses', 'Faint Sky Lenses'])
print("--- Faintness Constraints (True Lenses Only) ---")
for cat in lenses_only['faint_bin'].cat.categories:
    compute_subset_metrics(lenses_only, f"Profile: {cat}", f"faint_bin == '{cat}'")

# B. Arc Size Scale Breakdown
lenses_only['size_bin'] = pd.cut(lenses_only['arc_size'], bins=[0, 1.2, 1.5, 10.0], labels=['Small Scale (<1.2)', 'Medium Scale (1.2-1.5)', 'Large Scale (>1.5)'])
print("\n--- Structural Arc Scales (True Lenses Only) ---")
for cat in lenses_only['size_bin'].cat.categories:
    compute_subset_metrics(lenses_only, f"Scale: {cat}", f"size_bin == '{cat}'")

# C. Field Crowding / Blending Breakdown (Evaluated across the whole dataset)
print("\n--- Visual Field Blending (Full Dataset) ---")
compute_subset_metrics(analysis_df, "Isolated / Clean Fields", "blended == 0")
compute_subset_metrics(analysis_df, "Blended / Crowded Target Fields", "blended == 1")
