import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import xgboost as xgb
from cassava_root_model import RootVolumeDataset, DualPathRootVolumeModel, YOLOSegmentationModule, FeatureExtractor, CONFIG

# Ensure CUDA compatibility
import multiprocessing as mp
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

def load_xgboost_models(model_dir, num_folds):
    """Load pre-trained XGBoost models from disk."""
    xgb_models = []
    for fold in range(num_folds):
        model_path = os.path.join(model_dir, f"xgb_fold_{fold}.model")
        if os.path.exists(model_path):
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            xgb_models.append(model)
            print(f"Loaded XGBoost model for fold {fold} from {model_path}")
        else:
            raise FileNotFoundError(f"XGBoost model file not found: {model_path}")
    return xgb_models

def generate_cnn_predictions(test_loader, model_dir, device, num_folds):
    """Generate predictions using pre-trained CNN models across folds."""
    cnn_test_predictions = []
    ids = None
    
    for fold in range(num_folds):
        model = DualPathRootVolumeModel().to(device)
        model_path = os.path.join(model_dir, f"model_fold_{fold}.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"CNN model file not found: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Loaded CNN model for fold {fold} from {model_path}")
        
        fold_predictions = []
        with torch.no_grad():
            for images, features, id_vals in tqdm(test_loader, desc=f"CNN Fold {fold} Predictions"):
                images, features = images.to(device), features.to(device)
                outputs = model(images, features)
                fold_predictions.extend(outputs.cpu().numpy())
                if ids is None:  # Capture IDs only once
                    ids = id_vals
        cnn_test_predictions.append(fold_predictions)
    
    # Average predictions across folds
    cnn_test_preds = np.mean(np.array(cnn_test_predictions), axis=0)
    return ids, cnn_test_preds

def generate_xgboost_predictions(X_test, model_dir, num_folds):
    """Generate predictions using pre-trained XGBoost models across folds."""
    xgb_models = load_xgboost_models(model_dir, num_folds)
    xgb_test_predictions = []
    
    for fold, model in enumerate(xgb_models):
        preds = model.predict(X_test)
        xgb_test_predictions.append(preds)
        print(f"Generated XGBoost predictions for fold {fold}")
    
    # Average predictions across folds
    xgb_test_preds = np.mean(np.array(xgb_test_predictions), axis=0)
    return xgb_test_preds

def main():
    print(f"Using device: {CONFIG['device']}")
    
    # Load test data
    test_df = pd.read_csv(CONFIG["test_csv"])
    
    # Initialize YOLO segmentation module
    yolo_module = YOLOSegmentationModule(CONFIG["yolo_model_paths"])
    feature_extractor = FeatureExtractor(yolo_module)
    
    # Prepare test dataset
    test_dataset = RootVolumeDataset(test_df, CONFIG["data_dir"], is_train=False, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)
    
    # Extract test features for XGBoost
    test_features = []
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Extracting test features"):
        folder_name = row['FolderName']
        plant_number = row['PlantNumber']
        start_layer, end_layer = get_optimal_layers(folder_name, test_df)  # Use test_df as fallback
        
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
    feature_cols = [col for col in test_features_df.columns if col not in ['id']]
    X_test = test_features_df[feature_cols].values
    
    # Generate CNN predictions
    ids, cnn_test_preds = generate_cnn_predictions(test_loader, CONFIG["output_dir"], CONFIG["device"], CONFIG["num_folds"])
    
    # Generate XGBoost predictions
    xgb_test_preds = generate_xgboost_predictions(X_test, CONFIG["output_dir"], CONFIG["num_folds"])
    
    # Ensemble predictions (0.6 CNN + 0.4 XGBoost)
    ensemble_test_preds = 0.6 * cnn_test_preds + 0.4 * xgb_test_preds
    
    # Create submission file
    submission = pd.DataFrame({
        'ID': ids,
        'RootVolume': ensemble_test_preds
    })
    submission.to_csv(os.path.join(CONFIG["output_dir"], "submission.csv"), index=False)
    print(f"Submission file created at {os.path.join(CONFIG['output_dir'], 'submission.csv')}")

if __name__ == "__main__":
    main()