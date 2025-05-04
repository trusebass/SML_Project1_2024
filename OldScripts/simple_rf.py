"""
Simple RandomForest Pipeline for Distance Estimation
A stripped-down version with minimal preprocessing that focuses on
efficient and effective model training.
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
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

def main(model_params=None):
    """
    Main function to run the simple RandomForest pipeline.
    
    Args:
        model_params: Optional dictionary of model parameters to override defaults
    """
    start_time = time.time()
    print_progress("Starting simple RandomForest pipeline")
    
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
    img_dim = int(np.sqrt(train_images.shape[1]))
    print_progress(f"Original image dimensions: {img_dim}x{img_dim}")
    
    # Apply additional downsampling for speed if images are large
    if img_dim > 50 and 'downsample_factor' in config:
        print_progress(f"Further downsampling by factor of {config['downsample_factor']} for speed")
        # Reshaping for spatial downsampling
        n_samples = train_images.shape[0]
        factor = config['downsample_factor']
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
    
    # Create a simple pipeline with optimized RandomForest settings
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(
            random_state=42,
            n_jobs=4,           # Limit to 4 cores to avoid memory issues
            bootstrap=True,     # Enable bootstrapping for better stability
            oob_score=False,    # Disable OOB to save computation
            warm_start=False,   # Disable warm start for stability
            max_samples=0.7     # Use 70% of samples per tree for faster training
        ))
    ])
    
    # Default parameter grid - reduced size for faster and more stable training
    param_dist = {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__max_depth': [10, 20, None],
        'regressor__min_samples_split': [2, 5],
        'regressor__min_samples_leaf': [1, 2],
        'regressor__max_features': ['sqrt', 'log2']  # Removed 'auto' which can cause memory issues
    }
    
    # Calculate total parameter space size for logging
    param_space_size = 1
    for param, values in param_dist.items():
        param_space_size *= len(values)
    print_progress(f"Parameter space size: {param_space_size} combinations")
    
    # Default n_iter - randomized search will use at most this many iterations
    n_iter = min(15, param_space_size)
    print_progress(f"Using n_iter={n_iter} for RandomizedSearchCV")
    
    # Override with provided parameters if available
    if model_params:
        for param, value in model_params.items():
            if param.startswith('regressor__'):
                param_dist[param] = [value] if not isinstance(value, list) else value
            else:
                param_dist[f'regressor__{param}'] = [value] if not isinstance(value, list) else value
    
    # Use RandomizedSearchCV for efficient hyperparameter tuning
    print_progress("Starting hyperparameter search with RandomizedSearchCV")
    try:
        # First try with parallel processing but fewer CV folds for speed
        search = RandomizedSearchCV(
            pipeline, param_dist, n_iter=n_iter, cv=2,  # Reduced from 3 to 2 folds
            scoring='neg_mean_absolute_error', random_state=42, 
            n_jobs=4,  # Limit jobs to reduce memory pressure
            return_train_score=False,  # Saves memory and computation
            error_score='raise'  # Explicitly raise errors for debugging
        )
        
        # Train the model
        search.fit(train_images, train_distances)
    except Exception as e:
        print_progress(f"Error during parallel search: {str(e)}")
        print_progress("Trying again with minimal configuration...")
        
        # Try again with minimal configuration
        search = RandomizedSearchCV(
            pipeline, 
            {'regressor__n_estimators': [50],
             'regressor__max_depth': [10],
             'regressor__max_features': ['sqrt']}, 
            n_iter=1, cv=2,
            scoring='neg_mean_absolute_error', random_state=42, n_jobs=1,
            verbose=1  # Add verbosity to see progress
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
    model_name = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_SimpleRF.pkl"
    
    # Additional info for logging
    additional_info = {
        "model_type": "SimpleRandomForest",
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
    
    # # Save the model
    # models_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    # os.makedirs(models_folder, exist_ok=True)
    # model_path = os.path.join(models_folder, model_name)
    # print_progress(f"Saving model to {model_path}")
    # joblib.dump(best_model, model_path)
    # print_progress("Model saved successfully!")
    
    return mae, r2, total_time

if __name__ == "__main__":
    main()