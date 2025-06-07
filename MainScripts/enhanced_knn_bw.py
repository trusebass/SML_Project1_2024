#!/usr/bin/env python3
"""
Enhanced KNeighborsRegressor Pipeline for Grayscale Images
This script implements advanced image preprocessing techniques 
optimized for KNeighborsRegressor to estimate distances from grayscale images.
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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import BaseEstimator, TransformerMixin

# Image processing imports (allowed for preprocessing)
import cv2
from skimage.filters import sobel, gaussian, gabor, median, frangi, laplace
from skimage.exposure import equalize_hist, equalize_adapthist, rescale_intensity
from skimage.feature import local_binary_pattern, canny, hog
from skimage.morphology import disk, closing, opening, dilation


class EnhancedImagePreprocessor(BaseEstimator, TransformerMixin):
    """
    Enhanced image preprocessor with expanded preprocessing techniques for grayscale images
    """
    def __init__(self, use_edges=True, use_histogram_eq=True, use_clahe=True,
                 use_lbp=False, use_gabor=False, use_advanced_edges=False, 
                 use_morphology=False, use_hog=False, normalize_features=True):
        # Basic preprocessing options
        self.use_edges = use_edges
        self.use_histogram_eq = use_histogram_eq
        self.use_clahe = use_clahe
        
        # Advanced preprocessing options
        self.use_lbp = use_lbp
        self.use_gabor = use_gabor
        self.use_advanced_edges = use_advanced_edges
        self.use_morphology = use_morphology
        self.use_hog = use_hog
        
        # Additional options
        self.normalize_features = normalize_features
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        print(f"Enhanced preprocessing on {len(X)} images...")
        n_samples = len(X)
        
        # Calculate dimensions for grayscale images
        img_width = int(np.sqrt(X[0].shape[0]))
        img_height = img_width
        print(f"Detected grayscale images: {img_width}x{img_height}")
        
        # Initialize result list
        processed_features = []
        
        for i, flat_img in enumerate(X):
            if i % 100 == 0:
                print(f"Processing image {i}/{n_samples}")
            
            try:
                feature_list = []
                
                # Handle grayscale images
                img = flat_img.reshape(img_height, img_width)
                
                # Normalize if needed
                if img.max() > 1.0:
                    img = img.astype(np.float32) / 255.0
                
                # Always include original grayscale image
                feature_list.append(img.flatten())
                
                # Apply basic preprocessing
                self._add_basic_features(img, feature_list)
                
                # LBP features
                if self.use_lbp:
                    self._add_lbp_features(img, feature_list)
                
                # Gabor filter features
                if self.use_gabor:
                    self._add_gabor_features(img, feature_list)
                    
                # Advanced edge detection
                if self.use_advanced_edges:
                    self._add_advanced_edge_features(img, feature_list)
                    
                # Morphological operations
                if self.use_morphology:
                    self._add_morphology_features(img, feature_list)
                    
                # HOG features
                if self.use_hog:
                    self._add_hog_features(img, feature_list, img_width)
                
                # Combine all features
                combined_features = np.concatenate(feature_list)
                
                # Normalize the feature vector if requested
                if self.normalize_features:
                    feature_norm = np.linalg.norm(combined_features)
                    if feature_norm > 0:
                        combined_features = combined_features / feature_norm
                
                processed_features.append(combined_features)
                
            except Exception as e:
                print(f"Error processing image {i}: {str(e)}")
                # Fallback to original features
                processed_features.append(flat_img)
        
        return np.array(processed_features)
    
    def _add_basic_features(self, img, feature_list):
        """Add basic image processing features"""
        # Histogram equalization
        if self.use_histogram_eq:
            eq_img = equalize_hist(img)
            feature_list.append(eq_img.flatten())
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if self.use_clahe:
            clahe_img = equalize_adapthist(img, clip_limit=0.03)
            feature_list.append(clahe_img.flatten())
            
            # Additional intensity rescaling after CLAHE can help
            rescaled_img = rescale_intensity(clahe_img)
            feature_list.append(rescaled_img.flatten())
        
        # Edge detection with Sobel filter
        if self.use_edges:
            # Apply Gaussian blur to reduce noise
            blurred = gaussian(img, sigma=1)
            # Edge detection
            edges = sobel(blurred)
            feature_list.append(edges.flatten())
            
            # Try a different blur level
            blurred_strong = gaussian(img, sigma=2)
            edges_strong = sobel(blurred_strong)
            feature_list.append(edges_strong.flatten())
    
    def _add_lbp_features(self, img, feature_list):
        """Add Local Binary Pattern features"""
        # LBP is good for texture analysis
        # Parameters: P=8 (number of neighbors), R=1 (radius)
        try:
            lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
            feature_list.append(lbp.flatten())
            
            # Try a different radius
            lbp2 = local_binary_pattern(img, P=8, R=2, method='uniform')
            feature_list.append(lbp2.flatten())
        except:
            # If LBP fails (e.g., image too small), skip it
            pass
    
    def _add_gabor_features(self, img, feature_list):
        """Add Gabor filter features for texture detection"""
        try:
            # Apply Gabor filters at different frequencies and orientations
            for frequency in [0.1, 0.4]:
                for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                    gabor_real, gabor_imag = gabor(img, frequency=frequency, theta=theta, 
                                                  n_stds=3, mode='constant', cval=0)
                    # Use the real part which captures most of the texture information
                    feature_list.append(gabor_real.flatten())
        except:
            # If Gabor fails (e.g., image too small), skip it
            pass
    
    def _add_advanced_edge_features(self, img, feature_list):
        """Add features from different edge detection methods"""
        try:
            # Canny edge detection (more complex edge detector than Sobel)
            canny_edges = canny(img, sigma=1.5)
            feature_list.append(canny_edges.flatten())
            
            # Laplacian edge detection (detects zero crossings)
            laplacian_edges = laplace(img)
            feature_list.append(laplacian_edges.flatten())
            
            # Frangi filter for detecting ridge-like structures
            frangi_ridge = frangi(img)
            feature_list.append(frangi_ridge.flatten())
        except:
            pass
    
    def _add_morphology_features(self, img, feature_list):
        """Add features using morphological operations"""
        try:
            # Create a disk-shaped structuring element
            selem = disk(2)
            
            # Closing (dilation followed by erosion)
            closing_img = closing(img, selem)
            feature_list.append(closing_img.flatten())
            
            # Opening (erosion followed by dilation)
            opening_img = opening(img, selem)
            feature_list.append(opening_img.flatten())
            
            # Dilation (expands bright regions)
            dilated_img = dilation(img, selem)
            feature_list.append(dilated_img.flatten())
        except:
            pass
    
    def _add_hog_features(self, img, feature_list, img_width):
        """Add Histogram of Oriented Gradients features"""
        try:
            # Adjust HOG parameters based on image size
            if img_width >= 32:  # Larger images
                hog_feat = hog(img, orientations=8, pixels_per_cell=(8, 8), 
                               cells_per_block=(2, 2), visualize=False, feature_vector=True)
                feature_list.append(hog_feat)
            elif img_width >= 16:  # Medium images
                hog_feat = hog(img, orientations=6, pixels_per_cell=(4, 4),
                               cells_per_block=(2, 2), visualize=False, feature_vector=True)
                feature_list.append(hog_feat)
            # Skip for smaller images
        except:
            pass


def run_enhanced_knn_grid_search(X_train, y_train):
    """
    Run grid search for KNN with enhanced image preprocessing
    """
    # Define pipeline with preprocessing and KNN
    pipeline = Pipeline([
        ('preprocessor', EnhancedImagePreprocessor()),
        #('scaler', StandardScaler()),
        ('pca', PCA(random_state=42)),
        ('knn', KNeighborsRegressor())
    ])
    
    # Define parameter grid - include new preprocessing techniques
    param_grid = {
        # Basic preprocessing
        'preprocessor__use_edges': [True, False],
        'preprocessor__use_histogram_eq': [True, False],
        'preprocessor__use_clahe': [True, False],
        
        # Advanced preprocessing
        'preprocessor__use_lbp': [True, False],
        'preprocessor__use_gabor': [False, True],  # Slower, set to True if you want to try it
        'preprocessor__use_advanced_edges': [True, False],
        'preprocessor__use_morphology': [False, True],  # Set to True for more features
        'preprocessor__use_hog': [False, True],  # Set to True if images are large enough
        
        # Feature normalization
        'preprocessor__normalize_features': [True, False],
        
        # PCA and KNN parameters
        'pca__n_components': [150],
        'knn__n_neighbors': [3],
        'knn__weights': ['distance'],
        'knn__p': [2],  # Euclidean distance
    }
    
    # Create and run grid search
    print("Starting GridSearchCV...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=2,
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


def run_focused_knn(X_train, y_train):
    """
    Alternative to grid search - run a focused KNN with carefully selected parameters
    based on prior experiments that yielded MAE of 13.0
    """
    # Create a pipeline with enhanced preprocessing
    pipeline = Pipeline([
        ('preprocessor', EnhancedImagePreprocessor(
            use_edges=True,
            use_histogram_eq=True,
            use_clahe=True,
            use_lbp=False,
            use_gabor=False,
            use_advanced_edges=False,
            use_morphology=True,
            use_hog=True,
            normalize_features=False
        )),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=50, random_state=42)),
        ('knn', KNeighborsRegressor(n_neighbors=4, weights='distance', p=2))
    ])
    
    # Fit the pipeline
    print("Training focused KNN model...")
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    
    return pipeline



# ------ MAIN -----


if __name__ == "__main__":
    print("Starting Enhanced KNN Pipeline for Grayscale Images...")
    start_time = time.time()
    
    # Load configs from "config.yaml"
    config = load_config()
    
    # Make sure RGB loading is DISABLED for grayscale processing
    if config.get("load_rgb", False):
        print("[WARNING] RGB loading is enabled in config. Setting load_rgb=False for grayscale processing")
        config["load_rgb"] = False
    
    # Load dataset: images and corresponding minimum distance values
    images, distances, dataset = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")
    
    # For hyperparameter tuning phase: Use train/test split to evaluate performance
    FINAL_SUBMISSION = False  # Set to True for final model submission
    
    if not FINAL_SUBMISSION:
        # Development mode: Use train/test split for validation
        images_train, images_test, distances_train, distances_test = train_test_split(
            images, distances, test_size=0.2, random_state=42
        )
        
        # Use either grid search or focused training
        use_grid_search = False  # Set to False for faster focused training
        
        try:
            if use_grid_search:
                # Run grid search to find the best model
                best_model, best_params = run_enhanced_knn_grid_search(images_train, distances_train)
            else:
                # Run focused training with selected parameters
                best_model = run_focused_knn(images_train, distances_train)
            
            # Make predictions on test set
            pred_distances = best_model.predict(images_test)
            
            # Evaluate model
            mae, r2 = print_results(distances_test, pred_distances)
            
            # Visualize results
#            visualize_knn_results(distances_test, pred_distances)
            
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
            best_model = pipeline
    
    else:
        # Final submission mode: Train on ALL available data
        print("\n[INFO]: FINAL SUBMISSION MODE - Using all training data\n")
        
        try:
            # Train the model using all available training data
            final_model = run_focused_knn(images, distances)
            
            # # Save the model
            # models_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
            # os.makedirs(models_folder, exist_ok=True)
            
            # model_name = "grayscale_knn_full_training_model.pkl"
            # model_path = os.path.join(models_folder, model_name)
            # print(f"Saving final model trained on all data to {model_path}...")
            # joblib.dump(final_model, model_path)
            # print("Final model saved successfully!")
            
            # Load and predict on test data
            try:
                test_images = load_test_dataset(config)
                print(f"[INFO]: Test dataset loaded with {len(test_images)} samples.")
                
                # Make predictions on official test set
                test_predictions = final_model.predict(test_images)
                
                # Save predictions for submission
                save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "submission.csv")
                save_results(test_predictions)
                print(f"Saved predictions to submission.csv for competition submission")
                
            except Exception as e:
                print(f"Error during test prediction: {str(e)}")
                
        except Exception as e:
            print(f"Error during final model training: {str(e)}")
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")