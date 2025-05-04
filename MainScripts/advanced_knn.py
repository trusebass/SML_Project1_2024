#!/usr/bin/env python3
"""
Advanced KNeighborsRegressor Pipeline for Distance Estimation
This script implements sophisticated image preprocessing techniques 
optimized for KNeighborsRegressor to estimate distances from images.
"""

from utils import load_config, load_dataset, load_test_dataset, print_results, save_results

# Standard imports
import numpy as np
import joblib
import time
import os
import matplotlib.pyplot as plt

# scikit-learn imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import BaseEstimator, TransformerMixin

# Image processing imports (allowed for preprocessing)
import cv2
from skimage.feature import hog
from skimage.filters import sobel, gaussian
from skimage.exposure import equalize_hist, equalize_adapthist


class ImagePreprocessor(BaseEstimator, TransformerMixin):
    """
    Custom transformer for image preprocessing that combines multiple techniques
    to extract meaningful features from images.
    """
    def __init__(self, use_edges=True, use_hog=True, 
                 use_histogram_eq=True, use_clahe=True):
        self.use_edges = use_edges
        self.use_hog = use_hog
        self.use_histogram_eq = use_histogram_eq
        self.use_clahe = use_clahe
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        print(f"Preprocessing {len(X)} images...")
        n_samples = len(X)
        
        # Calculate image width - assume square images
        img_width = int(np.sqrt(X[0].shape[0]))
        print(f"Image width detected: {img_width}x{img_width} pixels")
        
        # Prepare for results
        processed_features = []
        
        for i, flat_img in enumerate(X):
            if i % 1000 == 0:
                print(f"Processing image {i}/{n_samples}")
                
            # Handle potential errors
            try:
                # Reshape flat array back to 2D image
                img = flat_img.reshape(img_width, img_width).astype(np.float64)
                
                # Normalize image to [0,1] range if not already
                if img.max() > 1.0:
                    img = img / 255.0
                
                # Create a feature vector for this image
                image_features = []
                
                # Apply histogram equalization if enabled
                if self.use_histogram_eq:
                    img_eq = equalize_hist(img)
                    image_features.append(img_eq.flatten())
                
                # Apply CLAHE if enabled
                if self.use_clahe:
                    img_clahe = equalize_adapthist(img, clip_limit=0.03)
                    image_features.append(img_clahe.flatten())
                
                # Extract edge features if enabled
                if self.use_edges:
                    # Apply Gaussian blur to reduce noise
                    img_blur = gaussian(img, sigma=1)
                    # Apply Sobel edge detection
                    edges = sobel(img_blur)
                    image_features.append(edges.flatten())
                
                # Extract HOG features if enabled and image is large enough
                if self.use_hog:
                    # Adjust HOG parameters based on image size
                    if img_width >= 32:  # Image is big enough for default params
                        hog_features = hog(
                            img, 
                            orientations=8, 
                            pixels_per_cell=(8, 8),  # Reduced from (16,16)
                            cells_per_block=(2, 2), 
                            visualize=False,
                            feature_vector=True
                        )
                        image_features.append(hog_features)
                    elif img_width >= 16:  # Smaller image needs adjusted params
                        hog_features = hog(
                            img, 
                            orientations=4,  # Fewer orientations
                            pixels_per_cell=(4, 4),  # Smaller cells
                            cells_per_block=(2, 2), 
                            visualize=False,
                            feature_vector=True
                        )
                        image_features.append(hog_features)
                    else:
                        # For very small images, skip HOG and use edges only
                        print(f"Image {i} too small for HOG ({img_width}x{img_width}), using edges only")
                
                # Always include the original image
                image_features.append(flat_img)
                
                # Concatenate all features
                combined_features = np.concatenate(image_features)
                processed_features.append(combined_features)
                
            except Exception as e:
                print(f"Error processing image {i}: {str(e)}")
                # In case of error, just use the original image features
                processed_features.append(flat_img)
                
        return np.array(processed_features)


def optimize_knn_model(X_train, y_train, cv=5):
    """
    Optimize KNN model using GridSearchCV with various preprocessing techniques
    and hyperparameters.
    """
    # Define the pipeline with preprocessing, dimensionality reduction, and KNN
    pipeline = Pipeline([
        ('preprocessor', ImagePreprocessor(use_edges=True, use_hog=True, 
                                          use_histogram_eq=True, use_clahe=True)),
        #('scaler', StandardScaler()),  # Change to StandardScaler for better stability
        ('pca', PCA(n_components=20, random_state=42)),
        ('knn', KNeighborsRegressor())
    ])
    
    # Define a simpler parameter grid with fewer combinations
    param_grid = {
        # Limit preprocessing combinations to reduce search space
        'preprocessor__use_edges': [True],
        'preprocessor__use_hog': [True],
        'preprocessor__use_histogram_eq': [True],
        'preprocessor__use_clahe': [True],
        
        # Fewer PCA components options
        'pca__n_components': [70, 90],
        
        # KNN parameters
        'knn__n_neighbors': [3, 7],
        'knn__weights': ['distance'],
        'knn__p': [2],  # Stick with Euclidean distance
    }
    
    # Create GridSearchCV object with error_score='raise' to debug fitting failures
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=2,
        error_score=np.nan  # Return NaN for failed fits rather than raising error
    )
    
    # Try first with a subset to confirm it works
    print("Starting GridSearchCV...")
    start_time = time.time()
    
    try:
        grid_search.fit(X_train, y_train)
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best MAE score: {-grid_search.best_score_:.4f}")
    except Exception as e:
        print(f"Grid search failed with error: {str(e)}")
        # Fall back to basic KNN with minimal preprocessing if grid search fails
        print("Falling back to basic KNN model...")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=20, random_state=42)),
            ('knn', KNeighborsRegressor(n_neighbors=5, weights='distance'))
        ])
        pipeline.fit(X_train, y_train)
        return pipeline, {'fallback': True}
    
    end_time = time.time()
    print(f"GridSearchCV completed in {end_time - start_time:.2f} seconds")
    
    return grid_search.best_estimator_, grid_search.best_params_


def visualize_results(y_true, y_pred, best_params):
    """
    Visualize model performance with scatter plot and error distribution.
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
    ax2.set_xlabel(f"{best_params}")
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Error Distribution (Mean: {np.mean(errors):.2f}, Std: {np.std(errors):.2f})')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('knn_results.png')
    print("Visualization saved to 'knn_results.png'")


if __name__ == "__main__":
    print("Starting Advanced KNN Pipeline...")
    start_time = time.time()
    
    # Load configs from "config.yaml"
    config = load_config()
    
    # Load dataset: images and corresponding minimum distance values
    images, distances, dataset = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")
    
    # Split data into training and testing sets
    images_train, images_test, distances_train, distances_test = train_test_split(
        images, distances, test_size=0.2, random_state=42
    )
    
    # Optimize KNN model
    best_model, best_params = optimize_knn_model(images_train, distances_train, cv=3)
    
    # Make predictions on test set
    pred_distances = best_model.predict(images_test)
    
    # Evaluate model
    mae, r2 = print_results(distances_test, pred_distances)
    
    # Visualize results
    visualize_results(distances_test, pred_distances, best_params)
    
    '''
    # Save the best model
    models_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_folder, exist_ok=True)
    
    model_name = "advanced_knn_model.pkl"
    model_path = os.path.join(models_folder, model_name)
    print(f"Saving model to {model_path}...")
    joblib.dump(best_model, model_path)
    print("Model saved successfully!")
    '''
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")