"""
Advanced Pipeline for Distance Estimation (targeting MAE < 8cm)
This implements a sophisticated approach to image-based distance estimation
with targeted optimizations for high accuracy while maintaining reasonable speed.
"""

from utils import (
    load_config,
    load_dataset,
    load_test_dataset,
    print_results,
    save_results,
    log_model_results,
)
import numpy as np
import os
import time
import datetime
import joblib
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.feature import hog
from skimage.filters import sobel
from skimage.measure import block_reduce
from scipy import ndimage

# Detect if running in Google Colab
def is_running_in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

IS_COLAB = is_running_in_colab()

def print_progress(message):
    """Simple progress printer that works in any environment"""
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}")
    sys.stdout.flush()

class MultiScaleFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract multi-scale image features optimized for distance estimation"""
    
    def __init__(self, target_sizes=[10, 20], use_hog=True, use_edges=True):
        self.target_sizes = target_sizes
        self.use_hog = use_hog
        self.use_edges = use_edges
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        n_samples = X.shape[0]
        orig_dim = int(np.sqrt(X.shape[1]))
        
        print_progress(f"Extracting multi-scale features from {n_samples} images...")
        
        # Preallocate list for feature arrays
        all_features = []
        
        # Process each image
        for i in range(n_samples):
            if i % 500 == 0 and i > 0:
                print_progress(f"Processed {i}/{n_samples} images")
            
            # Reshape to 2D
            img = X[i].reshape(orig_dim, orig_dim)
            img_features = []
            
            # 1. Multi-scale downsampling
            for size in self.target_sizes:
                # Skip if image is too small
                if orig_dim <= size:
                    small_img = img
                else:
                    # Calculate factors for efficient downscaling
                    factor = orig_dim // size
                    # Use block_reduce for efficient downscaling
                    small_img = block_reduce(img, (factor, factor), np.mean)
                
                # Add downscaled image features
                img_features.append(small_img.flatten())
                
                # Add edge information if requested
                if self.use_edges:
                    edges = sobel(small_img)
                    img_features.append(edges.flatten())
            
            # 2. HOG features (very effective for distance estimation)
            if self.use_hog:
                # Resize if needed to avoid HOG computation issues
                hog_size = min(orig_dim, 64)  # Cap HOG input size for speed
                if orig_dim != hog_size:
                    factor = orig_dim // hog_size
                    hog_img = block_reduce(img, (factor, factor), np.mean)
                else:
                    hog_img = img
                
                # Extract HOG features with parameters optimized for distance estimation
                hog_features = hog(
                    hog_img, 
                    orientations=8,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    feature_vector=True
                )
                img_features.append(hog_features)
            
            # 3. Statistical features (helpful for depth perception)
            # Calculate gradients in x and y directions for depth cues
            gx = ndimage.sobel(img, axis=0, mode='constant')
            gy = ndimage.sobel(img, axis=1, mode='constant')
            
            # Gradient magnitude and direction
            magnitude = np.sqrt(gx**2 + gy**2)
            
            # Statistical features from different parts of the image
            h, w = img.shape
            thirds_h = h // 3
            thirds_w = w // 3
            
            # Extract statistics from 9 regions (3x3 grid)
            for i in range(3):
                for j in range(3):
                    region = img[i*thirds_h:(i+1)*thirds_h, j*thirds_w:(j+1)*thirds_w]
                    region_mag = magnitude[i*thirds_h:(i+1)*thirds_h, j*thirds_w:(j+1)*thirds_w]
                    
                    # Add statistics from this region
                    stats = [
                        np.mean(region),      # Average intensity
                        np.std(region),       # Texture information
                        np.mean(region_mag),  # Average gradient (edge strength)
                        np.percentile(region, 75) - np.percentile(region, 25)  # Contrast measure
                    ]
                    img_features.append(np.array(stats))
            
            # Combine all features for this image
            all_features.append(np.concatenate(img_features))
        
        # Convert list to numpy array
        features_array = np.vstack([feat.reshape(1, -1) if feat.ndim == 1 else feat for feat in all_features])
        print_progress(f"Extracted {features_array.shape[1]} features per image")
        
        return features_array

def train_model_with_cross_validation(train_features, train_distances, n_splits=5):
    """Train multiple models and select the best one with cross validation"""
    
    print_progress(f"Training model with {n_splits}-fold cross-validation")
    
    # Create pipelines with different regressors
    pipelines = {
        "hist_gbm": Pipeline([
            ('scaler', QuantileTransformer(output_distribution='normal')),
            ('regressor', HistGradientBoostingRegressor(
                loss='absolute_error',  # Directly optimize for MAE
                random_state=42
            ))
        ]),
        
        "gbm": Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', GradientBoostingRegressor(
                loss='absolute_error',  # Directly optimize for MAE
                random_state=42
            ))
        ])
    }
    
    # Parameter distributions for random search
    param_dists = {
        "hist_gbm": {
            'regressor__learning_rate': [0.05, 0.1, 0.2],
            'regressor__max_depth': [3, 5, 7],
            'regressor__max_iter': [100, 200, 500],
            'regressor__min_samples_leaf': [5, 10, 20],
            'regressor__l2_regularization': [0, 0.1, 0.5, 1.0]
        },
        
        "gbm": {
            'regressor__learning_rate': [0.05, 0.1, 0.2],
            'regressor__n_estimators': [100, 200, 500],
            'regressor__max_depth': [3, 5, 7],
            'regressor__min_samples_leaf': [1, 3, 5],
            'regressor__subsample': [0.7, 0.8, 0.9, 1.0]
        }
    }
    
    # Separate some data for validation
    val_size = 0.2
    n_val = int(len(train_features) * val_size)
    train_idx = np.arange(len(train_features))
    np.random.seed(42)
    np.random.shuffle(train_idx)
    
    tr_idx = train_idx[n_val:]
    val_idx = train_idx[:n_val]
    
    X_tr = train_features[tr_idx]
    y_tr = train_distances[tr_idx]
    X_val = train_features[val_idx]
    y_val = train_distances[val_idx]
    
    # Train and evaluate each pipeline
    best_mae = float('inf')
    best_model = None
    best_params = None
    best_pipeline_name = None
    
    for name, pipeline in pipelines.items():
        print_progress(f"Training {name} with RandomizedSearchCV...")
        
        # Run randomized search
        search = RandomizedSearchCV(
            pipeline, 
            param_dists[name],
            n_iter=20,  # More iterations for better exploration
            cv=n_splits,
            scoring='neg_mean_absolute_error',
            random_state=42,
            n_jobs=-1
        )
        
        # Fit on training data
        search.fit(X_tr, y_tr)
        
        # Evaluate on validation set
        val_pred = search.predict(X_val)
        mae = np.mean(np.abs(val_pred - y_val))
        
        print_progress(f"{name} validation MAE: {mae:.4f}")
        
        if mae < best_mae:
            best_mae = mae
            best_model = search.best_estimator_
            best_params = search.best_params_
            best_pipeline_name = name
    
    print_progress(f"Best model: {best_pipeline_name} with validation MAE: {best_mae:.4f}")
    print_progress(f"Best parameters: {best_params}")
    
    # Return the best model
    return best_model, best_params, best_pipeline_name, best_mae

def analyze_errors(y_true, y_pred):
    """Analyze prediction errors and visualize them"""
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    
    # Create plots directory if it doesn't exist
    plot_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, alpha=0.7, color='blue')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title('Error Distribution')
    plt.xlabel('Prediction Error (cm)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, "error_distribution.png"))
    
    if IS_COLAB:
        plt.show()
    else:
        plt.close()
    
    # Plot absolute error vs true distance
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, abs_errors, alpha=0.5)
    plt.title('Absolute Error vs True Distance')
    plt.xlabel('True Distance (cm)')
    plt.ylabel('Absolute Error (cm)')
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(y_true, abs_errors, 1)
    p = np.poly1d(z)
    plt.plot(np.sort(y_true), p(np.sort(y_true)), "r--", alpha=0.8)
    
    plt.savefig(os.path.join(plot_dir, "error_vs_distance.png"))
    
    if IS_COLAB:
        plt.show()
    else:
        plt.close()
    
    # Calculate error statistics
    error_stats = {
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'mean_abs_error': np.mean(abs_errors),
        'median_abs_error': np.median(abs_errors),
        'std_error': np.std(errors),
        'max_abs_error': np.max(abs_errors),
        '90th_percentile_error': np.percentile(abs_errors, 90)
    }
    
    print_progress("\nError Analysis:")
    for stat, value in error_stats.items():
        print_progress(f"  {stat}: {value:.4f}")
    
    return error_stats

if __name__ == "__main__":
    start_time = time.time()
    print_progress("Starting advanced pipeline (targeting MAE < 8)")
    
    # Load config and dataset
    config = load_config()
    images, distances, dataset = load_dataset(config, "train")
    print_progress(f"Dataset loaded: {len(images)} samples")
    
    # Split data
    train_images, test_images, train_distances, test_distances = train_test_split(
        images, distances, test_size=0.2, random_state=42
    )
    
    # Extract advanced features
    feature_extractor = MultiScaleFeatureExtractor(
        target_sizes=[10, 20],  # Extract features at multiple scales
        use_hog=False,           # Use HOG features
        use_edges=False          # Include edge information
    )
    
    # Transform images to advanced feature representations
    train_features = feature_extractor.transform(train_images)
    test_features = feature_extractor.transform(test_images)
    
    feature_time = time.time()
    print_progress(f"Feature extraction completed in {feature_time - start_time:.2f} seconds")
    
    # Train model with cross-validation
    best_model, best_params, best_pipeline_name, val_mae = train_model_with_cross_validation(
        train_features, train_distances
    )
    
    training_time = time.time() - feature_time
    print_progress(f"Model training completed in {training_time:.2f} seconds")
    
    # Make predictions on test set
    print_progress("Generating predictions on test set...")
    pred_distances = best_model.predict(test_features)
    
    # Evaluate and print results
    mae, r2 = print_results(test_distances, pred_distances)
    
    # Analyze errors
    error_stats = analyze_errors(test_distances, pred_distances)
    
    total_time = time.time() - start_time
    print_progress(f"Total pipeline execution time: {total_time:.2f} seconds")
    
    # Log results
    model_name = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{best_pipeline_name}_Advanced.pkl"
    
    # Additional info for logging
    additional_info = {
        "model_type": f"Advanced_{best_pipeline_name}",
        "feature_method": "multi_scale_with_hog",
        "feature_count": train_features.shape[1],
        "dataset_size": len(images),
        "train_size": len(train_images),
        "test_size": len(test_images),
        "config_rgb": config["load_rgb"],
        "config_downsample": config["downsample_factor"],
        "validation_mae": val_mae,
        "error_stats": error_stats,
        "total_pipeline_time_seconds": total_time
    }
    
    # Log model results
    log_model_results(
        model_name=model_name,
        model_params=best_params,
        mae=mae,
        r2=r2,
        training_time=total_time,
        additional_info=additional_info
    )
    '''
    # Save the model
    models_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_folder, exist_ok=True)
    model_path = os.path.join(models_folder, model_name)
    print_progress(f"Saving model to {model_path}")
    joblib.dump(best_model, model_path)
    print_progress("Model saved successfully!")
    '''