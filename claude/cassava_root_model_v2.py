import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from tqdm import tqdm
import glob
from ultralytics import YOLO
import xgboost as xgb
import multiprocessing as mp

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

CONFIG = {
    "seed": 42,
    "train_csv": "data/Train.csv",
    "test_csv": "data/Test.csv",
    "data_dir": "data/data/",
    "model_dir": "data/Models/",
    "output_dir": "output/",
    "batch_size": 16,  # Increased for faster training
    "num_epochs": 100,  # More epochs with early stopping
    "learning_rate": 1e-4,
    "num_folds": 5,    # Reduced to 5 for simplicity
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "yolo_model_paths": {
        "early": "data/Models/best_early.pt",
        "late": "data/Models/best_late.pt",
        "full": "data/Models/best_full.pt",
    }
}

np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed(CONFIG["seed"])

os.makedirs(CONFIG["output_dir"], exist_ok=True)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((256, 256), dtype=np.float32)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.resize(img, (256, 256))
    return img.astype(np.float32) / 255.0

class RootVolumeDataset(Dataset):
    def __init__(self, df, data_dir, is_train=True, transform=None):
        self.df = df
        self.data_dir = data_dir
        self.is_train = is_train
        self.transform = transform
        self.yolo = YOLO(CONFIG["yolo_model_paths"]["full"]) if not is_train else None
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        folder_name = row['FolderName']
        plant_number = row['PlantNumber']
        start_layer, end_layer = int(row['Start']), int(row['End'])
        folder_path = os.path.join(self.data_dir, folder_name)
        
        # Load all images for both sides
        left_images, right_images = [], []
        areas = []
        for layer in range(start_layer, end_layer + 1):
            for side in ['L', 'R']:
                img_path = os.path.join(folder_path, f"*_{side}_{layer:03d}.png")
                img_files = glob.glob(img_path)
                if not img_files:
                    continue
                img = load_and_preprocess_image(img_files[0])
                if self.is_train:
                    mask_path = os.path.join(self.data_dir.replace('data/', 'train_labels/'), f"{folder_name}_{side}_{layer:03d}.txt")
                    if os.path.exists(mask_path):
                        with open(mask_path, 'r') as f:
                            lines = f.readlines()
                        if plant_number <= len(lines):
                            coords = [float(x) for x in lines[plant_number - 1].strip().split()[1:]]
                            coords = np.array(coords).reshape(-1, 2) * 256
                            mask = np.zeros((256, 256), dtype=np.uint8)
                            cv2.fillPoly(mask, [coords.astype(np.int32)], 1)
                            area = mask.sum() / (256 * 256)
                            if side == 'L':
                                left_images.append(img)
                            else:
                                right_images.append(img)
                            areas.append(area)
                else:
                    results = self.yolo.predict(img, conf=0.25, verbose=False)[0]
                    if results.masks and plant_number <= len(results.masks.data):
                        mask = results.masks.data[plant_number - 1].cpu().numpy()
                        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
                        area = mask.sum() / (256 * 256)
                        if side == 'L':
                            left_images.append(img)
                        else:
                            right_images.append(img)
                        areas.append(area)
        
        # Select best images or average
        best_left = np.mean(left_images, axis=0) if left_images else np.zeros((256, 256), dtype=np.float32)
        best_right = np.mean(right_images, axis=0) if right_images else np.zeros((256, 256), dtype=np.float32)
        area_feature = np.mean(areas) if areas else 0.0
        
        if self.transform and self.is_train:
            augmented = self.transform(image=best_left)
            best_left = augmented["image"]
            augmented = self.transform(image=best_right)
            best_right = augmented["image"]
        
        combined_image = torch.tensor(np.stack([best_left, best_right]), dtype=torch.float32)
        features = torch.tensor([area_feature, plant_number / 7.0], dtype=torch.float32)
        
        if self.is_train:
            target = torch.tensor(row['RootVolume'], dtype=torch.float32)
            return combined_image, features, target
        return combined_image, features, row['ID']

class DualPathRootVolumeModel(nn.Module):
    def __init__(self, feature_dim=2):
        super().__init__()
        self.image_features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc_combined = nn.Sequential(
            nn.Linear(256 + feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, images, features):
        img_feats = self.image_features(images).view(images.size(0), -1)
        combined = torch.cat([img_feats, features], dim=1)
        return self.fc_combined(combined)

def train_fold(train_loader, val_loader, model, optimizer, criterion, device, num_epochs, patience=10):
    best_val_rmse = float('inf')
    best_model_state = None
    patience_counter = 0
    
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
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return best_model_state, best_val_rmse

def train_xgboost_model(X_train, y_train, X_val, y_val, fold, output_dir):
    params = {
        'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'learning_rate': 0.03,
        'max_depth': 7, 'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'n_estimators': 1000
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=100)
    model.save_model(os.path.join(output_dir, f"xgb_fold_{fold}.model"))
    print(f"Saved XGBoost model for fold {fold}")
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
    
    train_transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.GaussianBlur(blur_limit=3, p=0.5),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        A.GridDistortion(p=0.3),
    ])

    
    kf = KFold(n_splits=CONFIG["num_folds"], shuffle=True, random_state=CONFIG["seed"])
    fold_rmse_scores = []
    oof_predictions = np.zeros(len(train_df))
    
    # Feature extraction
    print("Extracting features...")
    all_features = []
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Training features"):
        folder_name = row['FolderName']
        plant_number = row['PlantNumber']
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
        
        combined_features.update({
            'plant_number': plant_number / 7.0,
            'is_early': 1 if str(row.get('Stage', 'unknown')).lower() == 'early' else 0,
            'is_late': 1 if str(row.get('Stage', 'unknown')).lower() == 'late' else 0,
            'root_volume': row['RootVolume'],
            'id': row['ID']
        })
        all_features.append(combined_features)
    
    features_df = pd.DataFrame(all_features)
    feature_cols = [col for col in features_df.columns if col not in ['root_volume', 'id']]
    X = features_df[feature_cols].values
    y = features_df['root_volume'].values
    
    test_features = []
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Test features"):
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
        
        combined_features.update({
            'plant_number': plant_number / 7.0,
            'is_early': 0,
            'is_late': 0,
            'id': row['ID']
        })
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
        optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        best_model_state, val_rmse = train_fold(train_loader, val_loader, model, optimizer, criterion, CONFIG["device"], CONFIG["num_epochs"])
        torch.save(best_model_state, os.path.join(CONFIG["output_dir"], f"model_fold_{fold}.pth"))
        
        model.load_state_dict(best_model_state)
        model.eval()
        val_preds, val_indices = [], []
        with torch.no_grad():
            for i, (images, features, targets) in enumerate(val_loader):
                images, features = images.to(CONFIG["device"]), features.to(CONFIG["device"])
                outputs = model(images, features)
                val_preds.extend(outputs.cpu().numpy())
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
    
    ensemble_oof_predictions = 0.7 * oof_predictions + 0.3 * xgb_oof_predictions  # Adjusted weights
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
    
    ensemble_test_preds = 0.7 * cnn_test_preds + 0.3 * xgb_test_preds
    submission = pd.DataFrame({'ID': test_df['ID'], 'RootVolume': ensemble_test_preds})
    submission.to_csv(os.path.join(CONFIG["output_dir"], "submission_ensemble.csv"), index=False)
    print("Submission file created successfully!")

if __name__ == "__main__":
    main()