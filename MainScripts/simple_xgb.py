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
    
    feature_time = time.time()
    print_progress(f"Data preparation completed in {feature_time - start_time:.2f} seconds")
    
    # Create a simple pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,
            verbosity=0
        ))
    ])
    
    # Default parameter grid - will be used if model_params is None
    param_dist = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [3, 5, 7, 9],
        'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'regressor__subsample': [0.7, 0.8, 0.9, 1.0],
        'regressor__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'regressor__min_child_weight': [1, 3, 5, 7]
    }
    
    # Override with provided parameters if available
    if model_params:
        for param, value in model_params.items():
            if param.startswith('regressor__'):
                param_dist[param] = [value] if not isinstance(value, list) else value
            else:
                param_dist[f'regressor__{param}'] = [value] if not isinstance(value, list) else value
    
    # Use RandomizedSearchCV for efficient hyperparameter tuning
    print_progress("Starting hyperparameter search with RandomizedSearchCV")
    search = RandomizedSearchCV(
        pipeline, param_dist, n_iter=10, cv=3,
        scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1
    )
    
    # Train the model
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