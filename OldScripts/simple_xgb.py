"""
Simple XGBoost Pipeline for Distance Estimation
A stripped-down version with minimal preprocessing that focuses on
efficient and effective model training using XGBoost.
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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sys

# Try to import XGBoost
try:
    from xgboost import XGBRegressor
except ImportError:
    print("XGBoost not installed. Installing now...")
    import subprocess
    subprocess.call(["pip", "install", "xgboost"])
    from xgboost import XGBRegressor

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

def main(model_params=None):
    """
    Main function to run the simple XGBoost pipeline.
    
    Args:
        model_params: Optional dictionary of model parameters to override defaults
    """
    start_time = time.time()
    print_progress("Starting simple XGBoost pipeline")
    
    # Load config and dataset
    config = load_config()
    images, distances, dataset = load_dataset(config, "train")
    print_progress(f"Dataset loaded: {len(images)} samples")
    
    # Split data
    train_images, test_images, train_distances, test_distances = train_test_split(
        images, distances, test_size=0.2, random_state=42
    )
    
    # Optionally downsample the images for even faster training
    # This can be controlled via config.yaml's downsample_factor
    img_size = train_images.shape[1]
    img_dim = int(np.sqrt(img_size))
    print_progress(f"Original image size: {img_size}, trying to reshape to {img_dim}x{img_dim}")
    
    # Validate if image is perfectly square
    if img_dim * img_dim != img_size:
        print_progress(f"Warning: Image is not perfectly square ({img_size} != {img_dim}²={img_dim*img_dim})")
        # Find the closest perfect square dimension
        img_dim = int(np.sqrt(img_size))
        if img_dim * img_dim > img_size:
            img_dim -= 1
        print_progress(f"Adjusting to closest square dimension: {img_dim}x{img_dim} = {img_dim*img_dim}")
        # Truncate images to the closest perfect square
        if img_dim * img_dim < img_size:
            train_images = train_images[:, :img_dim*img_dim]
            test_images = test_images[:, :img_dim*img_dim]
            print_progress(f"Truncated images to {img_dim*img_dim} pixels")
    
    print_progress(f"Original image dimensions: {img_dim}x{img_dim}")
    
    # Apply additional downsampling for speed if images are large
    if img_dim > 50 and 'downsample_factor' in config:
        # Ensure downsample factor divides the dimension evenly
        factor = config['downsample_factor']
        if img_dim % factor != 0:
            # Find the closest factor that divides evenly
            valid_factors = [f for f in range(1, min(factor*2, img_dim+1)) if img_dim % f == 0]
            if valid_factors:
                closest_factor = min(valid_factors, key=lambda x: abs(x - factor))
                print_progress(f"Warning: Downsample factor {factor} doesn't divide {img_dim} evenly.")
                print_progress(f"Adjusting to closest valid factor: {closest_factor}")
                factor = closest_factor
            else:
                # Fallback to a safe value
                factor = 1
                print_progress(f"Warning: No valid downsample factors found. Using factor=1")
        
        print_progress(f"Further downsampling by factor of {factor} for speed")
        # Reshaping for spatial downsampling
        n_samples = train_images.shape[0]
        new_dim = img_dim // factor
        
        # Fast downsampling by averaging blocks
        downsampled_train = np.zeros((n_samples, new_dim * new_dim))
        for i in range(n_samples):
            img = train_images[i].reshape(img_dim, img_dim)
            # Simple block averaging for speed
            small_img = np.zeros((new_dim, new_dim))
            for y in range(new_dim):
                for x in range(new_dim):
                    y_start = y * factor
                    y_end = min((y + 1) * factor, img_dim)
                    x_start = x * factor
                    x_end = min((x + 1) * factor, img_dim)
                    small_img[y, x] = np.mean(img[y_start:y_end, x_start:x_end])
            downsampled_train[i] = small_img.flatten()
            
        # Same for test images
        n_test = test_images.shape[0]
        downsampled_test = np.zeros((n_test, new_dim * new_dim))
        for i in range(n_test):
            img = test_images[i].reshape(img_dim, img_dim)
            small_img = np.zeros((new_dim, new_dim))
            for y in range(new_dim):
                for x in range(new_dim):
                    y_start = y * factor
                    y_end = min((y + 1) * factor, img_dim)
                    x_start = x * factor
                    x_end = min((x + 1) * factor, img_dim)
                    small_img[y, x] = np.mean(img[y_start:y_end, x_start:x_end])
            downsampled_test[i] = small_img.flatten()
            
        # Replace original images with downsampled versions
        train_images = downsampled_train
        test_images = downsampled_test
        print_progress(f"Downsampled to {new_dim}x{new_dim} = {new_dim*new_dim} features")
        
    feature_time = time.time()
    print_progress(f"Data preparation completed in {feature_time - start_time:.2f} seconds")
    
    # Create a simple pipeline with optimized XGBoost settings
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=4,          # Limit to 4 cores instead of -1 to avoid memory issues
            verbosity=0,
            tree_method='hist', # Use histogram-based algorithm (much faster)
            grow_policy='lossguide', # More efficient tree growth
            max_bin=256        # Reduced bins for faster training
        ))
    ])
    
    # Default parameter grid - will be used if model_params is None
    param_dist = {
        'regressor__n_estimators': [100, 200, 500],
        'regressor__max_depth': [3, 7, None],
        'regressor__learning_rate': [0.01, 0.1, 0.2],
        'regressor__subsample': [0.8, 1.0],
        'regressor__colsample_bytree': [0.8, 1.0],
        'regressor__min_child_weight': [1, 5]
    }
    
    # Override with provided parameters if available
    if model_params:
        # Clear the original param_dist if model_params is provided
        param_dist = {}
        for param, value in model_params.items():
            param_key = f'regressor__{param}' if not param.startswith('regressor__') else param
            param_dist[param_key] = [value] if not isinstance(value, list) else value
    
    # Calculate total parameter space size for logging
    param_space_size = 1
    for param, values in param_dist.items():
        param_space_size *= len(values)
    print_progress(f"Parameter space size: {param_space_size} combinations")
    
    # Adjust n_iter to be at most the size of the parameter space to avoid warning
    n_iter = min(20, param_space_size)
    print_progress(f"Using n_iter={n_iter} for RandomizedSearchCV")
    
    # Use RandomizedSearchCV for efficient hyperparameter tuning
    print_progress("Starting hyperparameter search with RandomizedSearchCV")
    try:
        # First try with parallel processing but fewer CV folds for speed
        search = RandomizedSearchCV(
            pipeline, param_dist, n_iter=n_iter, cv=2,  # Reduced from 3 to 2 folds
            scoring='neg_mean_absolute_error', random_state=42, 
            n_jobs=4,  # Limit jobs to reduce memory pressure
            return_train_score=False  # Saves memory and computation
        )
        
        # Train the model
        search.fit(train_images, train_distances)
    except Exception as e:
        print_progress(f"Error during parallel search: {str(e)}")
        print_progress("Trying again with minimal configuration...")
        
        # Try again with minimal configuration
        search = RandomizedSearchCV(
            pipeline, 
            {'regressor__n_estimators': [100],
             'regressor__max_depth': [3, 7],
             'regressor__learning_rate': [0.1]}, 
            n_iter=2, cv=2,
            scoring='neg_mean_absolute_error', random_state=42, n_jobs=1
        )
        
        # Train the model with reduced settings
        search.fit(train_images, train_distances)
    
    # Get best model
    best_model = search.best_estimator_
    training_time = time.time() - feature_time
    print_progress(f"Model training completed in {training_time:.2f} seconds")
    print_progress(f"Best parameters: {search.best_params_}")
    
    # Make predictions
    pred_distances = best_model.predict(test_images)
    
    # Evaluate and print results
    mae, r2 = print_results(test_distances, pred_distances)
    total_time = time.time() - start_time
    print_progress(f"Total pipeline execution time: {total_time:.2f} seconds")
    
    # Log results
    model_name = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_SimpleXGB.pkl"
    
    # Additional info for logging
    additional_info = {
        "model_type": "SimpleXGBoost",
        "feature_count": train_images.shape[1],
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
    # models_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    # os.makedirs(models_folder, exist_ok=True)
    # model_path = os.path.join(models_folder, model_name)
    # print_progress(f"Saving model to {model_path}")
    # joblib.dump(best_model, model_path)
    # print_progress("Model saved successfully!")
    
    return mae, r2, total_time

if __name__ == "__main__":
    main()