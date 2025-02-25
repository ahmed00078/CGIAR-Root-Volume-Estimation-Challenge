import torch
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from cassava_root_model import CassavaRootDataset, CassavaRootVolumeNet
import multiprocessing as mp  # Import for start method

# Set multiprocessing start method to 'spawn' for CUDA compatibility
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

# Define paths
PROJECT_ROOT = Path.cwd()  # /teamspace/studios/this_studio/
DATA_DIR = PROJECT_ROOT / 'data'
TEST_CSV = DATA_DIR / 'Test.csv'
MODEL_PATH = PROJECT_ROOT / 'best_model.pth'
SUBMISSION_DIR = PROJECT_ROOT / 'submissions'
SUBMISSION_FILE = SUBMISSION_DIR / 'submission_gen.csv'

def verify_files():
    """Verify required files exist."""
    required_files = [
        TEST_CSV,
        DATA_DIR / 'data' / 'test',
        DATA_DIR / 'Models' / 'best_full.pt',
        MODEL_PATH
    ]
    for file in required_files:
        if not file.exists():
            raise FileNotFoundError(f"Required file not found: {file}")
        print(f"Found: {file}")
    print("✓ All required files verified")

def generate_submission(model, test_loader, output_file):
    """Generate submission.csv from test predictions with progress tracking."""
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    predictions = []
    total_batches = len(test_loader)
    
    with torch.no_grad():
        for batch_idx, (features, ids) in enumerate(tqdm(test_loader, desc="Generating Predictions", total=total_batches)):
            try:
                features = features.to(device)
                outputs = model(features)
                for id_val, pred in zip(ids, outputs.cpu().numpy()):
                    predictions.append({'ID': id_val, 'RootVolume': float(pred)})
                if batch_idx % 10 == 0 and batch_idx > 0:
                    tqdm.write(f"Processed {batch_idx}/{total_batches} batches")
            except Exception as e:
                tqdm.write(f"Error in batch {batch_idx}: {str(e)}")
                continue
    
    submission_df = pd.DataFrame(predictions)
    submission_df.to_csv(output_file, index=False)
    print(f"Submission file generated at: {output_file}")

def main():
    SUBMISSION_DIR.mkdir(exist_ok=True)
    
    print("Starting submission generation on Lightning AI...")
    
    verify_files()
    
    test_dataset = CassavaRootDataset(
        csv_file=TEST_CSV,
        data_dir=DATA_DIR,
        split='test'
    )
    print(f"Test dataset loaded with {len(test_dataset)} samples")
    
    batch_size = 4
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    print(f"Test dataloader created with {len(test_loader)} batches (batch size: {batch_size})")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CassavaRootVolumeNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    print(f"Pre-trained model loaded from {MODEL_PATH} on {device}")
    
    generate_submission(model, test_loader, SUBMISSION_FILE)

if __name__ == "__main__":
    main()