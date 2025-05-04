#!/usr/bin/env python3
"""
Simplified KNeighborsRegressor Pipeline for Distance Estimation
This script implements effective image preprocessing techniques 
optimized for KNeighborsRegressor with reduced computational requirements.
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import BaseEstimator, TransformerMixin

# Image processing imports (allowed for preprocessing)
import cv2
from skimage.filters import sobel, gaussian
from skimage.exposure import equalize_hist


class SimpleImagePreprocessor(BaseEstimator, TransformerMixin):
    """
    Simplified image preprocessor that focuses on edge detection and histogram equalization
    """
    def __init__(self, use_edges=True, use_histogram_eq=True):
        self.use_edges = use_edges
        self.use_histogram_eq = use_histogram_eq
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        print(f"Preprocessing {len(X)} images...")
        n_samples = len(X)
        
        # Calculate image width - assume square images
        img_width = int(np.sqrt(X[0].shape[0]))
        
        # Prepare for results - allocate all at once to save memory operations
        if self.use_edges and self.use_histogram_eq:
            # Original + edges + histogram equalization
            processed_features = np.zeros((n_samples, X.shape[1] * 3))
        elif self.use_edges or self.use_histogram_eq:
            # Original + one preprocessing technique
            processed_features = np.zeros((n_samples, X.shape[1] * 2))
        else:
            # Just use original features
            return X
        
        for i, flat_img in enumerate(X):
            if i % 100 == 0:
                print(f"Processing image {i}/{n_samples}")
                
            # Reshape flat array back to 2D image
            img = flat_img.reshape(img_width, img_width).astype(np.float64)
            
            # Normalize image to [0,1] range if not already
            if img.max() > 1.0:
                img = img / 255.0
            
            # Start with original features
            feature_list = [flat_img]
            
            # Apply histogram equalization if enabled
            if self.use_histogram_eq:
                img_eq = equalize_hist(img)
                feature_list.append(img_eq.flatten())
            
            # Extract edge features if enabled
            if self.use_edges:
                # Apply Gaussian blur to reduce noise
                img_blur = gaussian(img, sigma=1)
                # Apply Sobel edge detection
                edges = sobel(img_blur)
                feature_list.append(edges.flatten())
            
            # Concatenate features and store
            processed_features[i] = np.concatenate(feature_list)
            
        return processed_features


def run_knn_grid_search(X_train, y_train):
    """
    Run a simplified grid search for KNN with preprocessed image features
    """
    # Define pipeline with preprocessing and KNN
    pipeline = Pipeline([
        ('preprocessor', SimpleImagePreprocessor()),
        ('scaler', StandardScaler()),
        ('pca', PCA(random_state=42)),
        ('knn', KNeighborsRegressor())
    ])
    
    # Define parameter grid - much smaller than before
    param_grid = {
        'preprocessor__use_edges': [True],
        'preprocessor__use_histogram_eq': [True],
        'pca__n_components': [20, 50],
        'knn__n_neighbors': [5, 7, 9],
        'knn__weights': ['distance'],
    }
    
    # Create and run grid search
    print("Starting GridSearchCV...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
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


if __name__ == "__main__":
    print("Starting Simplified KNN Pipeline...")
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
    
    # Run grid search to find the best model
    try:
        best_model, best_params = run_knn_grid_search(images_train, distances_train)
        
        # Make predictions on test set
        pred_distances = best_model.predict(images_test)
        
        # Evaluate model
        mae, r2 = print_results(distances_test, pred_distances)
        
        # Create plot of actual vs. predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(distances_test, pred_distances, alpha=0.5)
        plt.plot([distances_test.min(), distances_test.max()], 
                [distances_test.min(), distances_test.max()], 'r--')
        plt.xlabel('True Distance (cm)')
        plt.ylabel('Predicted Distance (cm)')
        plt.title('KNN Regressor: Predicted vs Actual Distances')
        plt.grid(True)
        plt.savefig('knn_results.png')
        
        # Save the best model
        models_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        os.makedirs(models_folder, exist_ok=True)
        
        model_name = "knn_model.pkl"
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
            ('pca', PCA(n_components=20, random_state=42)),
            ('knn', KNeighborsRegressor(n_neighbors=5, weights='distance', p=2))
        ])
        
        # Fit and evaluate
        pipeline.fit(images_train, distances_train)
        pred_distances = pipeline.predict(images_test)
        print_results(distances_test, pred_distances)
        
        # Save fallback model
        models_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        os.makedirs(models_folder, exist_ok=True)
        joblib.dump(pipeline, os.path.join(models_folder, "knn_fallback_model.pkl"))
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")