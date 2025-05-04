#!/usr/bin/env python3
"""
RGB-Compatible KNeighborsRegressor Pipeline for Distance Estimation
This script implements image preprocessing techniques that properly handle RGB images 
for KNeighborsRegressor to estimate distances.
"""

from utils import load_config, load_dataset, load_test_dataset, print_results, save_results

# Standard imports
import numpy as np
import joblib
import time
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# scikit-learn imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import BaseEstimator, TransformerMixin

# Image processing imports (allowed for preprocessing)
import cv2
from skimage.filters import sobel, gaussian
from skimage.exposure import equalize_hist, equalize_adapthist
from skimage.color import rgb2gray


class RGBImagePreprocessor(BaseEstimator, TransformerMixin):
    """
    Image preprocessor that properly handles both RGB and grayscale images
    """
    def __init__(self, use_edges=True, use_histogram_eq=True, use_clahe=True,
                 include_grayscale=True, include_rgb=True):
        self.use_edges = use_edges
        self.use_histogram_eq = use_histogram_eq
        self.use_clahe = use_clahe
        self.include_grayscale = include_grayscale
        self.include_rgb = include_rgb
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        print(f"Preprocessing {len(X)} images...")
        n_samples = len(X)
        
        # Check if input is likely RGB (feature dimension is divisible by 3)
        self.is_rgb_input = (X[0].shape[0] % 3 == 0)
        
        if self.is_rgb_input:
            # Calculate dimensions for RGB images
            img_width = int(np.sqrt(X[0].shape[0] / 3))
            img_height = img_width
            channels = 3
            print(f"Detected RGB images: {img_width}x{img_height}x{channels}")
        else:
            # Calculate dimensions for grayscale images
            img_width = int(np.sqrt(X[0].shape[0]))
            img_height = img_width
            channels = 1
            print(f"Detected grayscale images: {img_width}x{img_height}")
        
        # Initialize result list
        processed_features = []
        
        for i, flat_img in enumerate(X):
            if i % 100 == 0:
                print(f"Processing image {i}/{n_samples}")
            
            try:
                feature_list = []
                
                if self.is_rgb_input:
                    # Reshape RGB image properly (height, width, channels)
                    img = flat_img.reshape(img_height, img_width, channels)
                    
                    # Always include original RGB features if requested
                    if self.include_rgb:
                        # Normalize if needed
                        if img.max() > 1.0:
                            normalized_rgb = img.astype(np.float32) / 255.0
                        else:
                            normalized_rgb = img.astype(np.float32)
                        
                        # Add RGB channels as features
                        feature_list.append(normalized_rgb.flatten())
                    
                    # Convert to grayscale for additional processing
                    if self.include_grayscale:
                        # Use proper RGB to grayscale conversion
                        gray_img = rgb2gray(img)
                        
                        # Add grayscale as features
                        feature_list.append(gray_img.flatten())
                        
                        # Process the grayscale version further
                        if self.use_histogram_eq:
                            eq_img = equalize_hist(gray_img)
                            feature_list.append(eq_img.flatten())
                        
                        if self.use_clahe:
                            clahe_img = equalize_adapthist(gray_img, clip_limit=0.03)
                            feature_list.append(clahe_img.flatten())
                        
                        if self.use_edges:
                            # Apply Gaussian blur to reduce noise before edge detection
                            blurred = gaussian(gray_img, sigma=1)
                            # Edge detection on grayscale
                            edges = sobel(blurred)
                            feature_list.append(edges.flatten())
                    
                    # Process each color channel separately if needed
                    for channel in range(channels):
                        channel_img = img[:,:,channel]
                        
                        if self.use_edges:
                            channel_edges = sobel(gaussian(channel_img, sigma=1))
                            feature_list.append(channel_edges.flatten())
                
                else:
                    # Handle grayscale images
                    img = flat_img.reshape(img_height, img_width)
                    
                    # Normalize if needed
                    if img.max() > 1.0:
                        img = img.astype(np.float32) / 255.0
                    
                    # Always include original grayscale image
                    feature_list.append(img.flatten())
                    
                    # Apply additional processing
                    if self.use_histogram_eq:
                        eq_img = equalize_hist(img)
                        feature_list.append(eq_img.flatten())
                    
                    if self.use_clahe:
                        clahe_img = equalize_adapthist(img, clip_limit=0.03)
                        feature_list.append(clahe_img.flatten())
                    
                    if self.use_edges:
                        # Apply Gaussian blur to reduce noise
                        blurred = gaussian(img, sigma=1)
                        # Edge detection
                        edges = sobel(blurred)
                        feature_list.append(edges.flatten())
                
                # Combine all features
                processed_features.append(np.concatenate(feature_list))
                
            except Exception as e:
                print(f"Error processing image {i}: {str(e)}")
                # Fallback to original features
                processed_features.append(flat_img)
        
        return np.array(processed_features)


def run_rgb_knn_grid_search(X_train, y_train):
    """
    Run grid search for KNN with RGB-compatible image preprocessing
    """
    # Define pipeline with preprocessing and KNN
    pipeline = Pipeline([
        ('preprocessor', RGBImagePreprocessor()),
        ('scaler', StandardScaler()),
        ('pca', PCA(random_state=42)),
        ('knn', KNeighborsRegressor())
    ])
    
    # Define parameter grid
    param_grid = {
        'preprocessor__use_edges': [True],
        'preprocessor__use_histogram_eq': [True],
        'preprocessor__use_clahe': [True, False],
        'preprocessor__include_grayscale': [True],
        'preprocessor__include_rgb': [True],
        'pca__n_components': [20, 50, 100],
        'knn__n_neighbors': [5, 7, 9],
        'knn__weights': ['distance'],
        'knn__p': [2],  # Euclidean distance
    }
    
    # Create and run grid search
    print("Starting GridSearchCV...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=2
    )
    
    # Fit the grid search
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()
    
    # Print results
    print(f"GridSearchCV completed in {end_time - start_time:.2f} seconds")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best MAE score: {-grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_


def visualize_knn_results(y_true, y_pred, filename='rgb_knn_results.png'):
    """
    Visualize model performance with scatter plot and error distribution
    """
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot of predicted vs actual values
    ax1.scatter(y_true, y_pred, alpha=0.5)
    ax1.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    ax1.set_xlabel('True Distance (cm)')
    ax1.set_ylabel('Predicted Distance (cm)')
    ax1.set_title('Predicted vs Actual Distances')
    ax1.grid(True)
    
    # Error distribution
    errors = y_pred - y_true
    ax2.hist(errors, bins=30, alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='r', linestyle='--')
    ax2.set_xlabel('Prediction Error (cm)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Error Distribution (Mean: {np.mean(errors):.2f}, Std: {np.std(errors):.2f})')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Visualization saved to '{filename}'")


if __name__ == "__main__":
    print("Starting RGB-Compatible KNN Pipeline...")
    start_time = time.time()
    
    # Load configs from "config.yaml"
    config = load_config()
    
    # Make sure RGB loading is enabled
    if not config.get("load_rgb", False):
        print("[WARNING] RGB loading is not enabled in config. Setting load_rgb=True")
        config["load_rgb"] = True
    
    # Load dataset: images and corresponding minimum distance values
    images, distances, dataset = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")
    
    # Split data into training and testing sets
    images_train, images_test, distances_train, distances_test = train_test_split(
        images, distances, test_size=0.2, random_state=42
    )
    
    # Run grid search to find the best model
    try:
        best_model, best_params = run_rgb_knn_grid_search(images_train, distances_train)
        
        # Make predictions on test set
        pred_distances = best_model.predict(images_test)
        
        # Evaluate model
        mae, r2 = print_results(distances_test, pred_distances)
        
        # Visualize results
        visualize_knn_results(distances_test, pred_distances)
        
        # Save the best model
        models_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        os.makedirs(models_folder, exist_ok=True)
        
        model_name = "rgb_knn_model.pkl"
        model_path = os.path.join(models_folder, model_name)
        print(f"Saving model to {model_path}...")
        joblib.dump(best_model, model_path)
        print("Model saved successfully!")
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        print("Falling back to basic KNN pipeline...")
        
        # Simple fallback pipeline with minimal preprocessing
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=50, random_state=42)),
            ('knn', KNeighborsRegressor(n_neighbors=7, weights='distance', p=2))
        ])
        
        # Fit and evaluate
        pipeline.fit(images_train, distances_train)
        pred_distances = pipeline.predict(images_test)
        print_results(distances_test, pred_distances)
        
        # Save fallback model
        models_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        os.makedirs(models_folder, exist_ok=True)
        joblib.dump(pipeline, os.path.join(models_folder, "rgb_knn_fallback_model.pkl"))
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")