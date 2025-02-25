import os
import zipfile
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from cassava_root_model import CassavaRootDataset, CassavaRootVolumeNet, train_model
import multiprocessing as mp

# Set multiprocessing start method to 'spawn' for CUDA compatibility
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

# Project root
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / 'data'
TRAIN_CSV = DATA_DIR / 'Train.csv'
TEST_CSV = DATA_DIR / 'Test.csv'
MODEL_PATH = PROJECT_ROOT / 'best_model.pth'
SUBMISSION_DIR = PROJECT_ROOT / 'submissions'

print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"DATA_DIR: {DATA_DIR}")
print(f"TRAIN_CSV: {TRAIN_CSV}")
print(f"TEST_CSV: {TEST_CSV}")
print(f"MODEL_PATH: {MODEL_PATH}")
print(f"SUBMISSION_DIR: {SUBMISSION_DIR}")

def unzip_data():
    for zip_file, extract_dir in [
        (DATA_DIR / 'data.zip', DATA_DIR / 'data'),
        (DATA_DIR / 'train_labels.zip', DATA_DIR / 'train_labels'),
        (DATA_DIR / 'Models.zip', DATA_DIR / 'Models')
    ]:
        if zip_file.exists() and not extract_dir.exists():
            print(f"Unzipping {zip_file} to {extract_dir}")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_dir.parent)
        elif not zip_file.exists():
            print(f"Warning: {zip_file} not found, assuming already unzipped")
        print(f"Verified: {extract_dir}")

def prepare_data():
    required_files = [TRAIN_CSV, TEST_CSV, DATA_DIR / 'data' / 'train', DATA_DIR / 'data' / 'test', DATA_DIR / 'Models' / 'best_full.pt']
    for file in required_files:
        if not file.exists():
            raise FileNotFoundError(f"Required file/directory not found: {file}")
        print(f"Found: {file}")
    print("✓ All required files found")

def inference(model, test_loader, output_file):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    predictions = []
    with torch.no_grad():
        for features, ids in test_loader:
            imgs, areas = features
            imgs, areas = imgs.to(device), areas.to(device)
            outputs = model((imgs, areas))
            for id_val, pred in zip(ids, outputs.cpu().numpy()):
                predictions.append({'ID': id_val, 'RootVolume': pred})
    
    pd.DataFrame(predictions).to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")

def main():
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    print("Starting process on Lightning AI...")
    
    unzip_data()
    prepare_data()
    
    df = pd.read_csv(TRAIN_CSV)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv(DATA_DIR / 'train_split.csv', index=False)
    val_df.to_csv(DATA_DIR / 'val_split.csv', index=False)
    print("Data split and saved")
    
    train_dataset = CassavaRootDataset(csv_file=DATA_DIR / 'train_split.csv', data_dir=DATA_DIR, split='train')
    val_dataset = CassavaRootDataset(csv_file=DATA_DIR / 'val_split.csv', data_dir=DATA_DIR, split='train')
    test_dataset = CassavaRootDataset(csv_file=TEST_CSV, data_dir=DATA_DIR, split='test')
    print("Datasets created")
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)
    print("Dataloaders created")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CassavaRootVolumeNet()
    model = model.to(device)
    print(f"Training on {device}")
    
    train_model(model, train_loader, val_loader, epochs=50)
    
    inference(model, test_loader, SUBMISSION_DIR / 'submission_improved.csv')

if __name__ == "__main__":
    # Initialiser le multiprocessing correctement
    mp.set_start_method('spawn', force=True)
    # N'exécuter main() qu'une seule fois
    main()