import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from cassava_root_model import CassavaRootDataset, CassavaRootVolumeNet
import multiprocessing as mp

# Set multiprocessing for CUDA compatibility
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

# Project root and paths (matching your run.py)
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / 'data'
TEST_CSV = DATA_DIR / 'Test.csv'
MODEL_PATH = PROJECT_ROOT / 'best_model_v2.pth'  # Adjust if it's 'best_model.pth'
SUBMISSION_DIR = PROJECT_ROOT / 'submissions'

def inference(model, test_loader, output_file):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    predictions = []
    with torch.no_grad():
        for features, ids in tqdm(test_loader, desc="Generating Predictions"):
            imgs, areas = features
            imgs, areas = imgs.to(device), areas.to(device)
            outputs = model((imgs, areas))
            for id_val, pred in zip(ids, outputs.cpu().numpy()):
                predictions.append({'ID': id_val, 'RootVolume': float(pred)})
    
    pd.DataFrame(predictions).to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")

def main():
    print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}. Please train the model first with run.py.")
    
    # Load test data
    test_dataset = CassavaRootDataset(csv_file=TEST_CSV, data_dir=DATA_DIR, split='test')
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)
    print("Test dataset and dataloader created")
    
    # Load pre-trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CassavaRootVolumeNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    print(f"Loaded pre-trained model from {MODEL_PATH} on {device}")
    
    # Generate predictions
    inference(model, test_loader, SUBMISSION_DIR / 'submission.csv')

if __name__ == "__main__":
    main()