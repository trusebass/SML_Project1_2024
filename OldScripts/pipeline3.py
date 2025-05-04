"""
Lightweight Pipeline for Distance Estimation
This implements a simple, fast approach to image-based distance estimation
with minimal preprocessing and efficient model training.
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
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import sys

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

def fast_downscale(images, target_size=None, random_pixels=None):
    """
    Efficiently extract features from images using either:
    1. Simple downscaling to a very small size, or
    2. Random pixel sampling (surprisingly effective for this task)
    
    Parameters:
    -----------
    images : array
        Input images (samples × pixels)
    target_size : int, optional
        Target size for downscaling (e.g., 10 means 10×10 = 100 pixels)
    random_pixels : int, optional
        Number of random pixels to sample (overrides target_size if provided)
        
    Returns:
    --------
    features : array
        Extracted features (samples × features)
    """
    n_samples = images.shape[0]
    
    if random_pixels is not None:
        # Randomly sample pixels (very fast approach)
        print_progress(f"Extracting {random_pixels} random pixels per image...")
        orig_dim = int(np.sqrt(images.shape[1]))
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        pixel_indices = rng.choice(images.shape[1], size=random_pixels, replace=False)
        return images[:, pixel_indices]
    
    elif target_size is not None:
        # Simple downscaling by averaging blocks of pixels
        print_progress(f"Downscaling images to {target_size}×{target_size}...")
        orig_dim = int(np.sqrt(images.shape[1]))
        
        # Skip if already small enough
        if orig_dim <= target_size:
            return images
            
        # Calculate block size for average pooling
        block_size = orig_dim // target_size
        features = np.zeros((n_samples, target_size * target_size))
        
        for i in range(n_samples):
            if i % 100 == 0 and i > 0:
                print_progress(f"Processed {i}/{n_samples} images")
                
            # Reshape to 2D image
            img = images[i].reshape(orig_dim, orig_dim)
            small_img = np.zeros((target_size, target_size))
            
            # Simple average pooling
            for y in range(target_size):
                for x in range(target_size):
                    y_start = y * block_size
                    y_end = min((y + 1) * block_size, orig_dim)
                    x_start = x * block_size
                    x_end = min((x + 1) * block_size, orig_dim)
                    small_img[y, x] = np.mean(img[y_start:y_end, x_start:x_end])
            
            # Flatten back to 1D
            features[i] = small_img.flatten()
            
        return features
    
    else:
        # No feature extraction, return original images
        return images

if __name__ == "__main__":
    start_time = time.time()
    print_progress("Starting lightweight pipeline")
    
    # Load config and dataset
    config = load_config()
    images, distances, dataset = load_dataset(config, "train")
    print_progress(f"Dataset loaded: {len(images)} samples")
    
    # Split data
    train_images, test_images, train_distances, test_distances = train_test_split(
        images, distances, test_size=0.2, random_state=42
    )
    
    # Extract features - either use random pixels or downscaling
    use_random_pixels = True  # Set to False to use downscaling instead
    
    if use_random_pixels:
        # Random pixel sampling (extremely fast)
        train_features = fast_downscale(train_images, random_pixels=200)
        test_features = fast_downscale(test_images, random_pixels=200)
    else:
        # Simple downscaling (also fast)
        train_features = fast_downscale(train_images, target_size=20)
        test_features = fast_downscale(test_images, target_size=20)
    
    feature_time = time.time()
    print_progress(f"Feature extraction completed in {feature_time - start_time:.2f} seconds")
    
    # Create a simple pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('dim_reduction', TruncatedSVD(n_components=50, random_state=42)),
        ('regressor', GradientBoostingRegressor(random_state=42))
    ])
    
    # Simple parameter grid
    param_dist = {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__max_depth': [3, 5, 7],
        'regressor__learning_rate': [0.05, 0.1, 0.2],
        'dim_reduction__n_components': [20, 50, 80]
    }
    
    # Use RandomizedSearchCV instead of GridSearchCV for speed
    print_progress("Starting hyperparameter search with RandomizedSearchCV")
    search = RandomizedSearchCV(
        pipeline, param_dist, n_iter=10, cv=3, 
        scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1
    )
    
    # Train the model
    search.fit(train_features, train_distances)
    
    # Get best model
    best_model = search.best_estimator_
    training_time = time.time() - feature_time
    print_progress(f"Model training completed in {training_time:.2f} seconds")
    print_progress(f"Best parameters: {search.best_params_}")
    
    # Make predictions
    pred_distances = best_model.predict(test_features)
    
    # Evaluate and print results
    mae, r2 = print_results(test_distances, pred_distances)
    total_time = time.time() - start_time
    print_progress(f"Total pipeline execution time: {total_time:.2f} seconds")
    
    # Log results
    model_name = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_FastGBM.pkl"
    
    # Additional info for logging
    additional_info = {
        "model_type": "LightweightGradientBoosting",
        "feature_method": "random_pixels" if use_random_pixels else "downscaling",
        "feature_count": train_features.shape[1],
        "dataset_size": len(images),
        "train_size": len(train_images),
        "test_size": len(test_images),
        "config_rgb": config["load_rgb"],
        "config_downsample": config["downsample_factor"],
        "randomized_search_best_score": search.best_score_,
        "total_pipeline_time_seconds": total_time
    }
    
    # Log model results
    log_model_results(
        model_name=model_name,
        model_params=search.best_params_,
        mae=mae,
        r2=r2,
        training_time=total_time,
        additional_info=additional_info
    )
    
    # Save the model
    models_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_folder, exist_ok=True)
    model_path = os.path.join(models_folder, model_name)
    print_progress(f"Saving model to {model_path}")
    joblib.dump(best_model, model_path)
    print_progress("Model saved successfully!")