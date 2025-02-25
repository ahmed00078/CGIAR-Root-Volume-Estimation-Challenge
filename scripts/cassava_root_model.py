import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO

class CassavaRootDataset(Dataset):
    def __init__(self, csv_file, data_dir, split='train', target_size=(224, 224)):
        self.data = pd.read_csv(csv_file)
        self.base_dir = Path(data_dir).resolve()
        self.data_dir = self.base_dir / 'data' / split
        self.labels_dir = self.base_dir / 'train_labels' if split == 'train' else None
        self.split = split
        self.target_size = target_size
        self.yolo_model = YOLO(str(self.base_dir / 'Models' / 'best_full.pt')) if split == 'test' else None

        # Filter valid training samples
        if split == 'train':
            valid_rows = []
            for idx, row in self.data.iterrows():
                folder, side, plant = row['FolderName'], row['Side'], row['PlantNumber']
                start, end = int(row['Start']), int(row['End'])
                for layer in range(start, end + 1):
                    label_file = self.labels_dir / f"{folder}_{side}_{layer:03d}.txt"
                    img_files = list((self.data_dir / folder).glob(f"*_{side}_{layer:03d}.png"))
                    if label_file.exists() and img_files and plant <= sum(1 for _ in open(label_file)):
                        valid_rows.append(row)
                        break
            self.data = pd.DataFrame(valid_rows).reset_index(drop=True)
            print(f"Filtered to {len(self.data)} valid training samples")

        print(f"Dataset initialized with:")
        print(f"Base directory: {self.base_dir}")
        print(f"Data directory: {self.data_dir}")
        print(f"Labels directory: {self.labels_dir if split == 'train' else 'N/A'}")
        print(f"Split: {split}")
        print(f"Target size: {target_size}")
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        if split == 'train' and self.labels_dir is not None and not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

    def load_from_labels(self, folder_name, side, plant_number, start_layer, end_layer):
        folder_path = self.data_dir / folder_name
        labels_path = self.labels_dir
        best_area = 0
        best_features = None
        
        for layer in range(start_layer, end_layer + 1):
            label_file = labels_path / f"{folder_name}_{side}_{layer:03d}.txt"
            img_files = list((self.data_dir / folder_name).glob(f"*_{side}_{layer:03d}.png"))
            if not label_file.exists() or not img_files:
                continue
            
            img_file = img_files[0]
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, self.target_size)
            
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            if plant_number > len(lines):
                continue
            
            poly_line = lines[plant_number - 1].strip().split()
            coords = [float(x) for x in poly_line[1:]]
            coords = np.array(coords).reshape(-1, 2)
            mask = np.zeros(self.target_size, dtype=np.uint8)
            coords_scaled = (coords * np.array([self.target_size[0], self.target_size[1]])).astype(np.int32)
            cv2.fillPoly(mask, [coords_scaled], 1)
            
            area = mask.sum()
            if area > best_area:
                best_area = area
                features = img.astype(np.float32)
                features = (features - features.mean()) / (features.std() + 1e-6)
                best_features = (features, area / (self.target_size[0] * self.target_size[1]))
        
        return best_features  # Should always be valid due to filtering

    def load_and_segment_scans(self, folder_name, start_layer, end_layer, plant_number):
        folder_path = self.data_dir / folder_name
        best_area = 0
        best_features = None
        
        for layer in range(start_layer, end_layer + 1):
            for side in ['L', 'R']:
                img_files = list(folder_path.glob(f"*_{side}_{layer:03d}.png"))
                if not img_files:
                    continue
                img = cv2.imread(str(img_files[0]), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, self.target_size)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                
                if self.yolo_model is None:
                    continue
                    
                results = self.yolo_model.predict(img_rgb, verbose=False)
                if results[0].masks is None or plant_number > len(results[0].masks.data):
                    continue
                
                mask = results[0].masks.data[plant_number - 1].cpu().numpy() if hasattr(results[0].masks.data[plant_number - 1], 'cpu') else results[0].masks.data[plant_number - 1]
                mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
                area = mask.sum()
                if area > best_area:
                    best_area = area
                    features = img.astype(np.float32)
                    features = (features - features.mean()) / (features.std() + 1e-6)
                    best_features = (features, area / (self.target_size[0] * self.target_size[1]))
        
        if best_features is None:
            print(f"Warning: No valid segmentation for plant {plant_number} in folder {folder_name}")
            return (np.zeros(self.target_size, dtype=np.float32), 0.0)
        return best_features

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        folder_name = row['FolderName']
        plant_number = int(row['PlantNumber'])
        
        try:
            if self.split == 'train':
                side = row['Side']
                features, area = self.load_from_labels(folder_name, side, plant_number, int(row['Start']), int(row['End']))
                target = torch.tensor(row['RootVolume'], dtype=torch.float32)
                features = torch.from_numpy(features).float().unsqueeze(0)
                return (features, torch.tensor(area, dtype=torch.float32)), target
            else:
                features, area = self.load_and_segment_scans(folder_name, int(row['Start']), int(row['End']), plant_number)
                features = torch.from_numpy(features).float().unsqueeze(0)
                return (features, torch.tensor(area, dtype=torch.float32)), row['ID']
        except Exception as e:
            print(f"Error processing index {idx}, FolderName: {folder_name}: {str(e)}")
            raise

    def __len__(self):
        return len(self.data)

class CassavaRootVolumeNet(nn.Module):
    def __init__(self, input_channels=1):
        super(CassavaRootVolumeNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 7 * 7 + 1, 512),  # +1 for area
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        imgs, areas = x  # Unpack tuple
        x = self.conv_layers(imgs)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, areas.unsqueeze(1)], dim=1)  # Concatenate area
        x = self.fc_layers(x)
        return x.squeeze()

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target):
        return torch.sqrt(nn.MSELoss()(pred, target))

def train_model(model, train_loader, val_loader, epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = RMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Added weight decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    best_val_loss = float('inf')
    
    # Import PROJECT_ROOT from the outer scope
    from pathlib import Path
    PROJECT_ROOT = Path.cwd()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            try:
                features, areas = inputs
                features, areas, targets = features.to(device), areas.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model((features, areas))
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                batch_count += 1
                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs} - Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}')
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        avg_train_loss = train_loss / batch_count if batch_count > 0 else float('inf')
        
        model.eval()
        val_loss = 0.0
        val_batch_count = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                try:
                    features, areas = inputs
                    features, areas, targets = features.to(device), areas.to(device), targets.to(device)
                    outputs = model((features, areas))
                    val_loss += criterion(outputs, targets).item()
                    val_batch_count += 1
                except Exception as e:
                    print(f"Error in validation: {str(e)}")
                    continue
        
        avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Average Training Loss: {avg_train_loss:.4f}')
        print(f'Average Validation Loss: {avg_val_loss:.4f}')
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), str(PROJECT_ROOT / 'best_model_v2.pth'))
            print(f'New best model saved with validation loss: {best_val_loss:.4f}')