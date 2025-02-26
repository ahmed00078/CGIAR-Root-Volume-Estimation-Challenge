import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from tqdm import tqdm
import glob
from ultralytics import YOLO
import xgboost as xgb
import multiprocessing as mp
from typing import List
from albumentations.core.composition import Compose

# Set multiprocessing for CUDA compatibility
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

# Configuration (adjusted paths to match your structure)
CONFIG = {
    "seed": 42,
    "train_csv": "data/Train.csv",  # Root directory
    "test_csv": "data/Test.csv",    # Root directory
    "data_dir": "data/data/",  # Adjusted to match your nested data structure
    "model_dir": "data/Models/",  # Adjusted to match your structure
    "output_dir": "output/",
    "batch_size": 8,
    "num_epochs": 30,  # Increased epochs for better training
    "learning_rate": 1e-4,
    "num_folds": 5,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "yolo_model_paths": {
        "early": "data/Models/best_early.pt",
        "late": "data/Models/best_late.pt",
        "full": "data/Models/best_full.pt",
    }
}

# Set random seeds for reproducibility
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed(CONFIG["seed"])

# Create output directories
os.makedirs(CONFIG["output_dir"], exist_ok=True)

# Utility Functions (unchanged)
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = img / 255.0
    return img

def get_optimal_layers(folder_name, df):
    folder_data = df[df['FolderName'] == folder_name]
    if folder_data.empty:
        return 1, 100
    start = folder_data['Start'].mode().iloc[0]
    end = folder_data['End'].mode().iloc[0]
    return start, end

def get_image_paths(folder_path, side, start_layer, end_layer):
    pattern = os.path.join(folder_path, f"*_{side}_{start_layer:03d}.png")
    return sorted(glob.glob(pattern))

class RootVolumeDataset(Dataset):
    def __init__(self, df, data_dir, is_train=True, transform=None):
        self.df = df
        self.data_dir = data_dir
        self.is_train = is_train
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        folder_name = row['FolderName']
        plant_number = row['PlantNumber']
        
        start_layer, end_layer = get_optimal_layers(folder_name, self.df)
        folder_path = os.path.join(self.data_dir, folder_name)
        
        left_images = []
        right_images = []
        
        for layer in range(start_layer, end_layer + 1):
            left_path = os.path.join(folder_path, f"*_L_{layer:03d}.png")
            right_path = os.path.join(folder_path, f"*_R_{layer:03d}.png")
            
            left_files = glob.glob(left_path)
            right_files = glob.glob(right_path)
            
            if left_files:
                img = load_and_preprocess_image(left_files[0])
                if img is not None:
                    left_images.append(img)
            
            if right_files:
                img = load_and_preprocess_image(right_files[0])
                if img is not None:
                    right_images.append(img)
        
        best_left = self.select_best_image(left_images) if left_images else np.zeros((256, 256))
        best_right = self.select_best_image(right_images) if right_images else np.zeros((256, 256))
        
        if self.transform:
            augmented = self.transform(image=best_left)
            best_left = augmented["image"]
            augmented = self.transform(image=best_right)
            best_right = augmented["image"]
        
        best_left = torch.tensor(best_left, dtype=torch.float32).unsqueeze(0)
        best_right = torch.tensor(best_right, dtype=torch.float32).unsqueeze(0)
        combined_image = torch.cat([best_left, best_right], dim=0)
        
        features = self.extract_features(folder_name, plant_number, row.get('Genotype', 'Unknown'), row.get('Stage', 'Unknown'))
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        if self.is_train:
            target = torch.tensor(row['RootVolume'], dtype=torch.float32)
            return combined_image, features_tensor, target
        else:
            return combined_image, features_tensor, row['ID']
    
    def select_best_image(self, images):
        if not images:
            return np.zeros((256, 256), dtype=np.float32)
        std_devs = [np.std(img) for img in images]
        best_idx = np.argmax(std_devs)
        best_img = cv2.resize(images[best_idx], (256, 256))
        return best_img.astype(np.float32)  # Ensure consistent data type
    
    def extract_features(self, folder_name, plant_number, genotype, stage):
        genotypes = ['TME419', 'TMEB419', 'IITA-TMS-IBA30572', 'TMS60444', 'Unknown']
        genotype_features = [1 if g == genotype else 0 for g in genotypes]
        stage_features = [1 if stage == 'Early' else 0, 1 if stage == 'Late' else 0]
        plant_num_normalized = (plant_number - 1) / 6
        features = genotype_features + stage_features + [plant_num_normalized]
        return features

class DualPathRootVolumeModel(nn.Module):
    def __init__(self, feature_dim=8):
        super(DualPathRootVolumeModel, self).__init__()
        self.image_features = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc_combined = nn.Sequential(
            nn.Linear(128 + feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        
    def forward(self, images, features):
        img_feats = self.image_features(images).view(images.size(0), -1)
        combined = torch.cat([img_feats, features], dim=1)
        output = self.fc_combined(combined)
        return output.squeeze(1)

class YOLOSegmentationModule:
    def __init__(self, model_paths):
        self.models = {}
        for key, path in model_paths.items():
            if os.path.exists(path):
                self.models[key] = YOLO(path)
            else:
                print(f"Warning: Model file {path} not found.")
        self.default_model = self.models.get("full", next(iter(self.models.values())) if self.models else None)
    
    def segment_roots(self, image_path, stage="unknown"):
        if not self.models:
            print("No YOLO models available for segmentation")
            return None
        model = self.models.get(stage.lower(), self.default_model)
        if model is None:
            return None
        results = model(image_path, conf=0.25)
        return results[0]
    
    def extract_root_features(self, results, plant_number):
        if results is None or not hasattr(results, 'boxes'):
            return {'area': 0, 'width': 0, 'height': 0, 'aspect_ratio': 0, 'confidence': 0}
        plant_boxes = [box for box in results.boxes if box.cls.item() == plant_number - 1]
        if not plant_boxes:
            return {'area': 0, 'width': 0, 'height': 0, 'aspect_ratio': 0, 'confidence': 0}
        largest_box = max(plant_boxes, key=lambda x: (x.xyxy[0][2] - x.xyxy[0][0]) * (x.xyxy[0][3] - x.xyxy[0][1]))
        x1, y1, x2, y2 = largest_box.xyxy[0].tolist()
        width = x2 - x1
        height = y2 - y1
        return {
            'area': width * height,
            'width': width,
            'height': height,
            'aspect_ratio': width / height if height > 0 else 0,
            'confidence': largest_box.conf.item()
        }

class FeatureExtractor:
    def __init__(self, yolo_module):
        self.yolo_module = yolo_module
    
    def extract_features(self, folder_path, plant_number, side, start_layer, end_layer, stage="unknown"):
        features = []
        for layer in range(start_layer, end_layer + 1):
            image_path = os.path.join(folder_path, f"*_{side}_{layer:03d}.png")
            image_files = glob.glob(image_path)
            if not image_files:
                continue
            results = self.yolo_module.segment_roots(image_files[0], stage)
            if results is None:
                continue
            layer_features = self.yolo_module.extract_root_features(results, plant_number)
            layer_features['layer'] = layer
            layer_features['layer_normalized'] = (layer - start_layer) / (end_layer - start_layer) if end_layer > start_layer else 0
            features.append(layer_features)
        return features
    
    def aggregate_features(self, features_list):
        if not features_list:
            return {
                'max_area': 0, 'mean_area': 0, 'sum_area': 0, 'max_confidence': 0,
                'mean_confidence': 0, 'max_width': 0, 'max_height': 0,
                'mean_aspect_ratio': 0, 'num_detected_layers': 0, 'detection_ratio': 0
            }
        areas = [f['area'] for f in features_list]
        confidences = [f['confidence'] for f in features_list]
        widths = [f['width'] for f in features_list]
        heights = [f['height'] for f in features_list]
        aspect_ratios = [f['aspect_ratio'] for f in features_list]
        return {
            'max_area': max(areas), 'mean_area': np.mean(areas), 'sum_area': np.sum(areas),
            'max_confidence': max(confidences), 'mean_confidence': np.mean(confidences),
            'max_width': max(widths), 'max_height': max(heights),
            'mean_aspect_ratio': np.mean(aspect_ratios), 'num_detected_layers': len(features_list),
            'detection_ratio': len(features_list) / (len(features_list) + 1)
        }

def train_fold(train_loader, val_loader, model, optimizer, criterion, device, num_epochs):
    best_val_rmse = float('inf')
    best_model_state = None
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, features, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images, features, targets = images.to(device), features.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images, features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, features, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images, features, targets = images.to(device), features.to(device), targets.to(device)
                outputs = model(images, features)
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        val_rmse = rmse(val_targets, val_preds)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val RMSE: {val_rmse:.4f}")
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = model.state_dict().copy()
    return best_model_state, best_val_rmse

def train_xgboost_model(X_train, y_train, X_val, y_val, fold, output_dir):
    params = {
        'objective': 'reg:squarederror', 
        'eval_metric': 'rmse', 
        'learning_rate': 0.05,
        'max_depth': 6, 
        'min_child_weight': 1, 
        'subsample': 0.8, 
        'colsample_bytree': 0.8,
        'n_estimators': 1000,
        'early_stopping_rounds': 50  # Move this parameter to params dictionary
    }
    model = xgb.XGBRegressor(**params)
    
    # Remove early_stopping_rounds from fit()
    model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    model_path = os.path.join(output_dir, f"xgb_fold_{fold}.model")
    model.save_model(model_path)
    print(f"Saved XGBoost model for fold {fold} to {model_path}")
    return model

def generate_predictions(test_loader, model, device):
    model.eval()
    predictions, ids = [], []
    with torch.no_grad():
        for images, features, id_vals in tqdm(test_loader, desc="Generating predictions"):
            images, features = images.to(device), features.to(device)
            outputs = model(images, features)
            predictions.extend(outputs.cpu().numpy())
            ids.extend(id_vals)
    return ids, predictions

def main():
    print(f"Using device: {CONFIG['device']}")
    train_df = pd.read_csv(CONFIG["train_csv"])
    test_df = pd.read_csv(CONFIG["test_csv"])
    
    yolo_module = YOLOSegmentationModule(CONFIG["yolo_model_paths"])
    feature_extractor = FeatureExtractor(yolo_module)
    
    train_transform = Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.5),  # Use GaussianBlur instead of GaussNoise
    ])


    
    kf = KFold(n_splits=CONFIG["num_folds"], shuffle=True, random_state=CONFIG["seed"])
    fold_rmse_scores = []
    oof_predictions = np.zeros(len(train_df))
    
    print("Extracting features...")
    all_features = []
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Extracting training features"):
        folder_name = row['FolderName']
        plant_number = row['PlantNumber']
        stage = row.get('Stage', 'unknown')
        start_layer, end_layer = get_optimal_layers(folder_name, train_df)
        folder_path = os.path.join(CONFIG["data_dir"], folder_name)
        
        left_features = feature_extractor.extract_features(folder_path, plant_number, 'L', start_layer, end_layer, str(row.get('Stage', 'unknown')))
        right_features = feature_extractor.extract_features(folder_path, plant_number, 'R', start_layer, end_layer, str(row.get('Stage', 'unknown')))
        
        left_agg = feature_extractor.aggregate_features(left_features)
        right_agg = feature_extractor.aggregate_features(right_features)
        
        combined_features = {}
        for key in left_agg:
            combined_features[f'left_{key}'] = left_agg[key]
            combined_features[f'right_{key}'] = right_agg[key]
            combined_features[f'combined_{key}'] = left_agg[key] + right_agg[key]
        
        combined_features['plant_number'] = plant_number
        combined_features['is_early'] = 1 if str(row.get('Stage', 'unknown')).lower() == 'early' else 0
        combined_features['is_late'] = 1 if str(row.get('Stage', 'unknown')).lower() == 'late' else 0
        combined_features['root_volume'] = row['RootVolume']
        combined_features['id'] = row['ID']
        all_features.append(combined_features)
    
    features_df = pd.DataFrame(all_features)
    feature_cols = [col for col in features_df.columns if col not in ['root_volume', 'id']]
    X = features_df[feature_cols].values
    y = features_df['root_volume'].values
    
    test_features = []
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Extracting test features"):
        folder_name = row['FolderName']
        plant_number = row['PlantNumber']
        start_layer, end_layer = get_optimal_layers(folder_name, train_df)
        folder_path = os.path.join(CONFIG["data_dir"], folder_name)
        
        left_features = feature_extractor.extract_features(folder_path, plant_number, 'L', start_layer, end_layer)
        right_features = feature_extractor.extract_features(folder_path, plant_number, 'R', start_layer, end_layer)
        
        left_agg = feature_extractor.aggregate_features(left_features)
        right_agg = feature_extractor.aggregate_features(right_features)
        
        combined_features = {}
        for key in left_agg:
            combined_features[f'left_{key}'] = left_agg[key]
            combined_features[f'right_{key}'] = right_agg[key]
            combined_features[f'combined_{key}'] = left_agg[key] + right_agg[key]
        
        combined_features['plant_number'] = plant_number
        combined_features['is_early'] = 0
        combined_features['is_late'] = 0
        combined_features['id'] = row['ID']
        test_features.append(combined_features)
    
    test_features_df = pd.DataFrame(test_features)
    X_test = test_features_df[feature_cols].values
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        print(f"\n--- Fold {fold+1}/{CONFIG['num_folds']} ---")
        train_subset = train_df.iloc[train_idx].reset_index(drop=True)
        val_subset = train_df.iloc[val_idx].reset_index(drop=True)
        
        train_dataset = RootVolumeDataset(train_subset, CONFIG["data_dir"], is_train=True, transform=train_transform)
        val_dataset = RootVolumeDataset(val_subset, CONFIG["data_dir"], is_train=True, transform=None)
        
        train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)
        
        model = DualPathRootVolumeModel().to(CONFIG["device"])
        optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
        criterion = nn.MSELoss()
        
        best_model_state, val_rmse = train_fold(train_loader, val_loader, model, optimizer, criterion, CONFIG["device"], CONFIG["num_epochs"])
        torch.save(best_model_state, os.path.join(CONFIG["output_dir"], f"model_fold_{fold}.pth"))
        
        model.load_state_dict(best_model_state)
        model.eval()
        val_preds, val_targets, val_indices = [], [], []
        with torch.no_grad():
            for i, (images, features, targets) in enumerate(val_loader):
                images, features = images.to(CONFIG["device"]), features.to(CONFIG["device"])
                outputs = model(images, features)
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(targets.numpy())
                val_indices.extend(val_idx[i*CONFIG["batch_size"]:min((i+1)*CONFIG["batch_size"], len(val_subset))])
        for idx, pred in zip(val_indices, val_preds):
            oof_predictions[idx] = pred
        fold_rmse_scores.append(val_rmse)
        print(f"Fold {fold+1} RMSE: {val_rmse:.4f}")
    
    xgb_oof_predictions = np.zeros(len(train_df))
    xgb_fold_models = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n--- XGBoost Fold {fold+1}/{CONFIG['num_folds']} ---")
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        xgb_model = train_xgboost_model(X_train_fold, y_train_fold, X_val_fold, y_val_fold, fold, CONFIG["output_dir"])
        xgb_fold_models.append(xgb_model)
        val_preds = xgb_model.predict(X_val_fold)
        for i, idx in enumerate(val_idx):
            xgb_oof_predictions[idx] = val_preds[i]
        fold_rmse = rmse(y_val_fold, val_preds)
        print(f"XGBoost Fold {fold+1} RMSE: {fold_rmse:.4f}")
    
    ensemble_oof_predictions = 0.6 * oof_predictions + 0.4 * xgb_oof_predictions
    ensemble_rmse = rmse(y, ensemble_oof_predictions)
    print(f"\nCNN Mean RMSE: {np.mean(fold_rmse_scores):.4f}")
    print(f"XGBoost OOF RMSE: {rmse(y, xgb_oof_predictions):.4f}")
    print(f"Ensemble OOF RMSE: {ensemble_rmse:.4f}")
    
    test_dataset = RootVolumeDataset(test_df, CONFIG["data_dir"], is_train=False, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)
    
    cnn_test_predictions = []
    for fold in range(CONFIG["num_folds"]):
        model = DualPathRootVolumeModel().to(CONFIG["device"])
        model.load_state_dict(torch.load(os.path.join(CONFIG["output_dir"], f"model_fold_{fold}.pth")))
        ids, preds = generate_predictions(test_loader, model, CONFIG["device"])
        cnn_test_predictions.append(preds)
    cnn_test_preds = np.mean(np.array(cnn_test_predictions), axis=0)
    
    xgb_test_predictions = []
    for model in xgb_fold_models:
        xgb_test_predictions.append(model.predict(X_test))
    xgb_test_preds = np.mean(np.array(xgb_test_predictions), axis=0)
    
    ensemble_test_preds = 0.6 * cnn_test_preds + 0.4 * xgb_test_preds
    submission = pd.DataFrame({'ID': test_df['ID'], 'RootVolume': ensemble_test_preds})
    submission.to_csv(os.path.join(CONFIG["output_dir"], "submission_ensemble.csv"), index=False)
    print("Submission file created successfully!")

if __name__ == "__main__":
    main()