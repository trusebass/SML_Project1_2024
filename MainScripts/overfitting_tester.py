#!/usr/bin/env python3
"""
Overfitting Tester for Distance Estimation Models

This script tests a trained model for signs of overfitting by:
1. Testing on completely separate validation data
2. Comparing performance on augmented vs. original data
3. Analyzing error patterns and distribution
4. Performing cross-validation with different splits
"""

from utils import load_config, load_dataset, load_test_dataset, print_results
import numpy as np
import joblib
import os
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def load_model(model_path):
    """Load a trained model from disk"""
    print(f"Loading model from {model_path}...")
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None


def test_on_original_vs_augmented(model, images, distances, augmentation_factor=3):
    """
    Test model performance on original data vs. augmented data
    to detect if model performs suspiciously better on augmented data
    
    Returns:
        tuple: (original_mae, augmented_mae)
    """
    from augmented_knn import ImageAugmenter
    
    # Split data into original test set and augmentation test set
    orig_images, aug_source_images, orig_dist, aug_source_dist = train_test_split(
        images, distances, test_size=0.5, random_state=42
    )
    
    # Create augmented test data
    augmenter = ImageAugmenter(
        rotation_range=15,
        brightness_range=0.3,
        flip_horizontal=True,
        add_noise=True
    )
    
    aug_images, aug_dist = augmenter.augment_dataset(
        aug_source_images, aug_source_dist, 
        augmentation_factor=augmentation_factor
    )
    
    # Remove original images from augmented set to test only on new augmentations
    aug_images = aug_images[len(aug_source_images):]
    aug_dist = aug_dist[len(aug_source_dist):]
    
    # Test on original data
    print("\nTesting on original unaugmented data...")
    orig_pred = model.predict(orig_images)
    orig_mae = mean_absolute_error(orig_dist, orig_pred)
    orig_r2 = r2_score(orig_dist, orig_pred)
    
    print(f"Original data MAE: {orig_mae:.4f}")
    print(f"Original data R²: {orig_r2:.4f}")
    
    # Test on augmented data
    print("\nTesting on newly augmented data...")
    aug_pred = model.predict(aug_images)
    aug_mae = mean_absolute_error(aug_dist, aug_pred)
    aug_r2 = r2_score(aug_dist, aug_pred)
    
    print(f"Augmented data MAE: {aug_mae:.4f}")
    print(f"Augmented data R²: {aug_r2:.4f}")
    
    # Analyze difference
    diff = orig_mae - aug_mae
    print(f"\nMAE difference (orig - aug): {diff:.4f}")
    
    if diff > 2.0:
        print("WARNING: Model performs much better on augmented data!")
        print("This suggests that augmentation may be creating artificial patterns.")
    elif diff < -2.0:
        print("WARNING: Model performs much worse on augmented data!")
        print("This suggests your augmentation might be too aggressive.")
    else:
        print("Model performance is reasonably consistent between original and augmented data.")
    
    return orig_mae, aug_mae


def cross_validation_test(model, images, distances, n_splits=5):
    """
    Perform cross-validation to test model stability across different data splits
    """
    print("\nPerforming cross-validation...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    mae_scores = []
    r2_scores = []
    fold_size = len(images) // n_splits
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(images)):
        # Use only test indices
        X_test = images[test_idx]
        y_test = distances[test_idx]
        
        print(f"Testing fold {fold+1}/{n_splits} with {len(test_idx)} samples...")
        
        # Predict on this fold's test data
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        fold_mae = mean_absolute_error(y_test, y_pred)
        fold_r2 = r2_score(y_test, y_pred)
        
        print(f"  Fold {fold+1} MAE: {fold_mae:.4f}, R²: {fold_r2:.4f}")
        
        mae_scores.append(fold_mae)
        r2_scores.append(fold_r2)
    
    mae_mean = np.mean(mae_scores)
    mae_std = np.std(mae_scores)
    r2_mean = np.mean(r2_scores)
    r2_std = np.std(r2_scores)
    
    print(f"\nCross-validation results:")
    print(f"MAE: {mae_mean:.4f} ± {mae_std:.4f}")
    print(f"R²: {r2_mean:.4f} ± {r2_std:.4f}")
    
    # Check for consistency
    if mae_std > 1.5:
        print("\nWARNING: High variance in MAE across folds!")
        print("This suggests the model may be overfitting or very sensitive to the data split.")
    else:
        print("\nMAE is relatively consistent across different data splits.")
    
    return mae_mean, mae_std


def analyze_error_distribution(model, images, distances):
    """
    Analyze the distribution of prediction errors to detect patterns
    """
    print("\nAnalyzing error distribution...")
    
    # Get predictions
    predictions = model.predict(images)
    
    # Calculate errors
    errors = predictions - distances
    abs_errors = np.abs(errors)
    
    # Sort distances to analyze error vs distance
    sorted_indices = np.argsort(distances)
    sorted_distances = distances[sorted_indices]
    sorted_abs_errors = abs_errors[sorted_indices]
    
    # Group errors by distance ranges
    dist_ranges = [
        (0, 100),
        (100, 200),
        (200, 300),
        (300, 400),
        (400, 500)
    ]
    
    print("\nError analysis by distance range:")
    for low, high in dist_ranges:
        mask = (distances >= low) & (distances < high)
        if np.sum(mask) > 0:
            range_errors = abs_errors[mask]
            range_mae = np.mean(range_errors)
            print(f"  Distance {low}-{high} cm: MAE = {range_mae:.4f}, samples = {np.sum(mask)}")
    
    # Plot error distribution
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Error histogram
    plt.subplot(2, 2, 1)
    plt.hist(errors, bins=30, color='skyblue', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', linewidth=1)
    plt.title('Prediction Error Distribution')
    plt.xlabel('Error (predicted - actual)')
    plt.ylabel('Frequency')
    
    # Plot 2: Absolute error vs actual distance
    plt.subplot(2, 2, 2)
    plt.scatter(distances, abs_errors, alpha=0.5, s=10)
    plt.title('Error vs. Actual Distance')
    plt.xlabel('Actual Distance (cm)')
    plt.ylabel('Absolute Error (cm)')
    
    # Add smoothed trendline
    from scipy.ndimage import gaussian_filter1d
    # Calculate moving average trendline
    window_size = max(10, len(sorted_distances) // 50)  # Adaptive window size
    smoothed_errors = gaussian_filter1d(sorted_abs_errors, window_size)
    plt.plot(sorted_distances, smoothed_errors, 'r-', linewidth=2)
    
    # Plot 3: Predicted vs actual scatter plot
    plt.subplot(2, 2, 3)
    plt.scatter(distances, predictions, alpha=0.5, s=10)
    min_val = min(np.min(distances), np.min(predictions))
    max_val = max(np.max(distances), np.max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')  # Perfect prediction line
    plt.title('Predicted vs. Actual Distance')
    plt.xlabel('Actual Distance (cm)')
    plt.ylabel('Predicted Distance (cm)')
    
    # Plot 4: Distribution of actual vs. predicted
    plt.subplot(2, 2, 4)
    plt.hist(distances, bins=30, alpha=0.5, label='Actual', color='blue')
    plt.hist(predictions, bins=30, alpha=0.5, label='Predicted', color='green')
    plt.title('Distribution: Actual vs. Predicted')
    plt.xlabel('Distance (cm)')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('overfitting_analysis.png')
    print(f"\nError analysis plots saved to 'overfitting_analysis.png'")
    
    # Check for signs of overfitting in the error distribution
    if np.mean(errors) > 1.0:
        print("\nWARNING: Predictions are biased (tend to overestimate)")
    elif np.mean(errors) < -1.0:
        print("\nWARNING: Predictions are biased (tend to underestimate)")
    
    # Check if errors are larger for certain distance ranges
    max_range_mae = 0
    min_range_mae = float('inf')
    for low, high in dist_ranges:
        mask = (distances >= low) & (distances < high)
        if np.sum(mask) > 10:  # Only consider ranges with enough samples
            range_mae = np.mean(abs_errors[mask])
            max_range_mae = max(max_range_mae, range_mae)
            min_range_mae = min(min_range_mae, range_mae)
    
    if max_range_mae > 2 * min_range_mae:
        print("\nWARNING: Error varies significantly across distance ranges")
        print("The model may be overfitting to certain distance ranges")
    
    return np.mean(abs_errors), np.std(errors)


if __name__ == "__main__":
    print("Starting Overfitting Analysis...")
    start_time = time.time()
    
    # Load configs from "config.yaml"
    config = load_config()
    
    # Select which model to analyze
    print("\nAvailable models:")
    models_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    model_files = [f for f in os.listdir(models_folder) if f.endswith('.pkl')]
    
    for i, model_file in enumerate(model_files):
        print(f"[{i+1}] {model_file}")
    
    try:
        selection = int(input("\nEnter the number of the model to analyze: "))
        selected_model = model_files[selection - 1]
    except (ValueError, IndexError):
        print("Invalid selection. Using the first model.")
        selected_model = model_files[0] if model_files else "model.pkl"
    
    model_path = os.path.join(models_folder, selected_model)
    
    # Load the model
    model = load_model(model_path)
    if model is None:
        print("Failed to load model. Exiting.")
        exit(1)
    
    # Determine if the model expects RGB images based on the model filename
    use_rgb = 'rgb' in selected_model.lower() and 'gray' not in selected_model.lower()
    
    # Set config appropriately
    if use_rgb:
        config["load_rgb"] = True
        print("Using RGB images based on model name")
    else:
        config["load_rgb"] = False
        print("Using grayscale images based on model name")
    
    # Load dataset
    print("\nLoading dataset...")
    images, distances, dataset = load_dataset(config)
    print(f"Dataset loaded with {len(images)} samples.")
    
    # Perform overfitting tests
    print("\n" + "=" * 50)
    print("OVERFITTING TEST 1: CROSS-VALIDATION")
    cross_val_mae, cross_val_std = cross_validation_test(model, images, distances)
    
    print("\n" + "=" * 50)
    print("OVERFITTING TEST 2: ORIGINAL VS AUGMENTED DATA")
    orig_mae, aug_mae = test_on_original_vs_augmented(model, images, distances, 
                                                    augmentation_factor=3)
    
    print("\n" + "=" * 50)
    print("OVERFITTING TEST 3: ERROR DISTRIBUTION ANALYSIS")
    overall_mae, error_std = analyze_error_distribution(model, images, distances)
    
    # Overfitting score calculation (higher = more likely overfit)
    print("\n" + "=" * 50)
    print("OVERALL OVERFITTING ASSESSMENT")
    
    overfitting_score = 0
    
    # Factor 1: Cross-validation stability (high std = unstable = possible overfit)
    if cross_val_std > 2.0:
        overfitting_score += 3
    elif cross_val_std > 1.0:
        overfitting_score += 2
    elif cross_val_std > 0.5:
        overfitting_score += 1
    
    # Factor 2: Original vs augmented performance gap
    aug_orig_diff = orig_mae - aug_mae
    if aug_orig_diff > 3.0:
        overfitting_score += 3
        print("- Model performs MUCH better on augmented data than original data")
    elif aug_orig_diff > 1.5:
        overfitting_score += 2
        print("- Model performs better on augmented data than original data")
    elif aug_orig_diff > 0.5:
        overfitting_score += 1
        print("- Model performs slightly better on augmented data")
    
    # Factor 3: Overall performance seems too good to be true
    if overall_mae < 6.0:
        overfitting_score += 3
        print("- Overall MAE is suspiciously low (<6)")
    elif overall_mae < 8.0:
        overfitting_score += 1
        print("- Overall MAE is quite low (<8)")
    
    # Calculate overfitting risk
    risk_level = "LOW"
    if overfitting_score >= 6:
        risk_level = "VERY HIGH"
    elif overfitting_score >= 4:
        risk_level = "HIGH"
    elif overfitting_score >= 2:
        risk_level = "MODERATE"
    
    print(f"\nOverfitting risk assessment:")
    print(f"Risk score: {overfitting_score}/9")
    print(f"Risk level: {risk_level}")
    
    if risk_level == "LOW" or risk_level == "MODERATE":
        print("\nRECOMMENDATION: Your model shows acceptable resilience to overfitting.")
    else:
        print("\nRECOMMENDATION: Your model shows signs of overfitting. Consider:")
        print("1. Reducing augmentation factor")
        print("2. Adding regularization to your pipeline")
        print("3. Using less aggressive preprocessing")
        print("4. Checking for data leakage between train/test")
    
    end_time = time.time()
    print(f"\nAnalysis completed in {end_time - start_time:.2f} seconds")