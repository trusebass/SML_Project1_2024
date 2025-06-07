#!/usr/bin/env python3
"""
Augmented KNeighborsRegressor Pipeline for Distance Estimation
This script implements data augmentation techniques to expand the training dataset
along with advanced image preprocessing for improved distance estimation accuracy.
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
from skimage import transform, exposure, util
from skimage.filters import sobel, gaussian
from skimage.exposure import equalize_hist, equalize_adapthist, rescale_intensity
from skimage.feature import local_binary_pattern


class ImageAugmenter:
    """
    Class for augmenting images to artificially increase dataset size
    """
    def __init__(self, rotation_range=10, brightness_range=0.2, 
                 flip_horizontal=True, add_noise=True):
        self.rotation_range = rotation_range  # degrees
        self.brightness_range = brightness_range  # ratio
        self.flip_horizontal = flip_horizontal
        self.add_noise = add_noise
    
    def augment_dataset(self, images, distances, augmentation_factor=2):
        """
        Augment the dataset by the specified factor
        
        Args:
            images: Original image dataset (each row is a flattened image)
            distances: Original distance labels
            augmentation_factor: How many times to multiply the dataset (can be float, e.g., 2.5)
            
        Returns:
            augmented_images, augmented_distances
        """
        print(f"Augmenting dataset by factor {augmentation_factor}...")
        
        if images.shape[0] == 0 or distances.shape[0] == 0:
            print("Empty dataset provided, cannot augment")
            return images, distances
            
        # Determine if images are likely RGB based on dimensions
        first_img_size = images[0].shape[0]
        img_dim_if_rgb = int(np.sqrt(first_img_size / 3))
        is_color = (first_img_size == img_dim_if_rgb * img_dim_if_rgb * 3)
        
        # Calculate image dimensions based on whether RGB or grayscale
        if is_color:
            img_dim = img_dim_if_rgb
            channels = 3
        else:
            img_dim = int(np.sqrt(first_img_size))  # For grayscale
            channels = 1
            
        print(f"Detected {'RGB' if is_color else 'grayscale'} images: {img_dim}x{img_dim} with {channels} channel(s)")
        
        # Create arrays to hold augmented data
        n_original = images.shape[0]
        
        # Initialize with original data
        augmented_images = [images]
        augmented_distances = [distances]
        
        # Initialize random number generator for reproducibility
        rng = np.random.RandomState(42)
        
        # Calculate how many new augmented batches to create (integer part)
        full_batches = int(augmentation_factor - 1)
        
        # Calculate the fractional part for partial batch
        fraction = augmentation_factor - 1 - full_batches
        
        # Generate full augmented batches
        for i in range(full_batches):
            print(f"Creating full augmentation batch {i+1}/{full_batches}")
            augmented_batch = []
            
            # Process each image in the dataset
            for j, flat_img in enumerate(tqdm(images, desc=f"Augmenting batch {i+1}")):
                try:
                    # Create augmented image
                    aug_img = self._augment_single_image(flat_img, img_dim, channels, is_color, rng)
                    augmented_batch.append(aug_img)
                    
                except Exception as e:
                    print(f"Error augmenting image {j}: {str(e)}. Using original.")
                    # If augmentation fails, use the original image instead
                    augmented_batch.append(flat_img)
            
            # Add augmented batch to full augmented dataset
            augmented_images.append(np.array(augmented_batch))
            augmented_distances.append(distances)  # Same distances for augmented images
        
        # Create partial batch if needed (for fractional augmentation factor)
        if fraction > 0:
            print(f"Creating partial augmentation batch ({fraction:.2f} of original size)")
            partial_batch = []
            
            # Calculate how many images to augment in this partial batch
            n_partial = int(n_original * fraction)
            
            # Randomly select indices to augment
            indices_to_augment = rng.choice(n_original, size=n_partial, replace=False)
            
            # Process only selected images
            for idx in tqdm(indices_to_augment, desc="Creating partial batch"):
                try:
                    # Create augmented image
                    aug_img = self._augment_single_image(images[idx], img_dim, channels, is_color, rng)
                    partial_batch.append(aug_img)
                    
                except Exception as e:
                    print(f"Error augmenting image {idx}: {str(e)}. Using original.")
                    # If augmentation fails, use the original image instead
                    partial_batch.append(images[idx])
            
            # Add partial batch to augmented dataset
            augmented_images.append(np.array(partial_batch))
            augmented_distances.append(distances[indices_to_augment])
        
        # Concatenate all augmented data
        final_images = np.vstack(augmented_images)
        final_distances = np.concatenate(augmented_distances)
        
        print(f"Dataset augmented from {n_original} to {final_images.shape[0]} samples")
        print(f"Achieved augmentation factor: {final_images.shape[0] / n_original:.2f}x")
        
        return final_images, final_distances
    
    def _augment_single_image(self, flat_img, img_dim, channels, is_color, rng):
        """Helper method to augment a single image"""
        # Reshape to 2D/3D
        if is_color:
            img = flat_img.reshape(img_dim, img_dim, channels)
        else:
            img = flat_img.reshape(img_dim, img_dim)  # For grayscale, only 2D
        
        # Ensure proper normalization before processing
        if img.max() > 1.0:
            img = img.astype(np.float32) / 255.0
        
        # Ensure img is in range [0,1] for processing
        img = np.clip(img, 0, 1)
        
        # Copy image to avoid modifying original
        aug_img = img.copy()
        
        # Apply random augmentations
        # 1. Random rotation
        if rng.random() > 0.5:
            angle = rng.uniform(-self.rotation_range, self.rotation_range)
            aug_img = transform.rotate(aug_img, angle, mode='edge', preserve_range=True)
        
        # 2. Random brightness adjustment
        if rng.random() > 0.5:
            factor = rng.uniform(1.0 - self.brightness_range, 1.0 + self.brightness_range)
            aug_img = exposure.adjust_gamma(aug_img, factor)
            # Ensure values stay in valid range after brightness adjustment
            aug_img = np.clip(aug_img, 0, 1)
        
        # 3. Horizontal flip
        if self.flip_horizontal and rng.random() > 0.7:
            aug_img = np.fliplr(aug_img)
        
        # 4. Add noise
        if self.add_noise and rng.random() > 0.7:
            # Set noise intensity based on image range but keep it conservative
            intensity = rng.uniform(0.005, 0.02)
            # Use mode 'gaussian' for noise
            aug_img = util.random_noise(aug_img, mode='gaussian', var=intensity, clip=True)
        
        # Ensure final values are in valid range
        aug_img = np.clip(aug_img, 0, 1)
        
        # Convert back to original value range if needed
        if flat_img.max() > 1.0:
            aug_img = (aug_img * 255).astype(flat_img.dtype)
        
        # Flatten and return
        return aug_img.flatten()


class ImagePreprocessor(BaseEstimator, TransformerMixin):
    """
    Image preprocessor that properly handles both RGB and grayscale images
    """
    def __init__(self, use_edges=True, use_histogram_eq=True, use_clahe=True, use_lbp=False, 
                 force_grayscale=False, use_gabor=False, use_advanced_edges=False, 
                 use_morphology=False, use_hog=False):
        self.use_edges = use_edges
        self.use_histogram_eq = use_histogram_eq
        self.use_clahe = use_clahe
        self.use_lbp = use_lbp
        self.force_grayscale = force_grayscale  # Use this to force grayscale processing
        self.use_gabor = use_gabor
        self.use_advanced_edges = use_advanced_edges
        self.use_morphology = use_morphology
        self.use_hog = use_hog
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        print(f"Preprocessing {len(X)} images...")
        n_samples = len(X)
        
        # Check if we should force grayscale mode
        if self.force_grayscale:
            self.is_rgb_input = False
            img_width = int(np.sqrt(X[0].shape[0]))
            img_height = img_width
            channels = 1
            print(f"Forced grayscale mode: {img_width}x{img_height}")
        else:
            # Determine if images are likely RGB based on dimensions
            first_img_size = X[0].shape[0]
            img_dim_if_rgb = int(np.sqrt(first_img_size / 3))
            
            # Verify if it's actually RGB by checking if the dimensions match up
            self.is_rgb_input = (first_img_size == img_dim_if_rgb * img_dim_if_rgb * 3)
            
            if self.is_rgb_input:
                # Calculate dimensions for RGB images
                img_width = img_dim_if_rgb
                img_height = img_width
                channels = 3
                print(f"Detected RGB images: {img_width}x{img_height}x{channels}")
            else:
                # Calculate dimensions for grayscale images
                img_width = int(np.sqrt(first_img_size))
                img_height = img_width
                channels = 1
                print(f"Detected grayscale images: {img_width}x{img_height}")
        
        # Initialize result list
        processed_features = []
        
        # First, determine the expected feature dimensions to ensure consistency
        # Process the first image to get expected feature dimension
        try:
            sample_features = self._process_single_image(X[0], img_height, img_width, channels)
            expected_dim = len(sample_features)
            print(f"Expected feature dimension: {expected_dim}")
        except Exception as e:
            print(f"Error determining feature dimensions: {str(e)}")
            # Use original images if preprocessing fails completely
            return X
        
        # Process all images with consistent dimensions
        for i, flat_img in enumerate(X):
            if i % 200 == 0:
                print(f"Processing image {i}/{n_samples}")
            
            try:
                # Process the image
                features = self._process_single_image(flat_img, img_height, img_width, channels)
                
                # Check if dimensions match expected
                if len(features) == expected_dim:
                    processed_features.append(features)
                else:
                    print(f"Warning: Inconsistent feature dimensions for image {i}. Using original.")
                    # If dimensions don't match, use original to maintain consistency
                    processed_features.append(flat_img)
                    
            except Exception as e:
                print(f"Error processing image {i}: {str(e)}")
                # Fallback to original features
                processed_features.append(flat_img)
        
        try:
            # Convert to numpy array all at once to ensure consistent shapes
            result = np.array(processed_features)
            print(f"Final feature array shape: {result.shape}")
            return result
        except ValueError as e:
            print(f"Error creating feature array: {str(e)}")
            print("Falling back to original images")
            return X
    
    def _process_single_image(self, flat_img, img_height, img_width, channels):
        """Process a single image and return its feature vector"""
        feature_list = []
        
        if self.is_rgb_input:
            # Reshape RGB image properly (height, width, channels)
            img = flat_img.reshape(img_height, img_width, channels)
            
            # Normalize if needed
            if img.max() > 1.0:
                img = img.astype(np.float32) / 255.0
            
            # Always include original RGB features
            feature_list.append(img.flatten())
            
            # Convert to grayscale for additional processing
            # Simple average of channels for grayscale conversion
            gray_img = np.mean(img, axis=2)
            feature_list.append(gray_img.flatten())
            
            # Apply preprocessing to grayscale version
            self._add_features(gray_img, feature_list)
            
            # Process each color channel separately
            for channel in range(channels):
                channel_img = img[:,:,channel]
                if self.use_edges:
                    # Edge detection on each channel
                    edges = sobel(gaussian(channel_img, sigma=1))
                    feature_list.append(edges.flatten())
        
        else:
            # Handle grayscale images
            img = flat_img.reshape(img_height, img_width)
            
            # Normalize if needed
            if img.max() > 1.0:
                img = img.astype(np.float32) / 255.0
            
            # Always include original grayscale image
            feature_list.append(img.flatten())
            
            # Apply preprocessing
            self._add_features(img, feature_list)
        
        # Combine all features
        return np.concatenate(feature_list)
    
    def _add_features(self, img, feature_list):
        """Add image preprocessing features"""
        try:
            # Ensure image is properly scaled for processing
            img_norm = np.clip(img, 0, 1)
            
            # Histogram equalization
            if self.use_histogram_eq:
                eq_img = equalize_hist(img_norm)
                feature_list.append(eq_img.flatten())
            
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            if self.use_clahe:
                clahe_img = equalize_adapthist(img_norm, clip_limit=0.03)
                feature_list.append(clahe_img.flatten())
                
                # Additional intensity rescaling can help
                rescaled_img = rescale_intensity(clahe_img)
                feature_list.append(rescaled_img.flatten())
            
            # Edge detection with Sobel filter
            if self.use_edges:
                # Apply Gaussian blur to reduce noise
                blurred = gaussian(img_norm, sigma=1)
                # Edge detection
                edges = sobel(blurred)
                feature_list.append(edges.flatten())
            
            # LBP features for texture analysis
            if self.use_lbp:
                lbp = local_binary_pattern(img_norm, P=8, R=1, method='uniform')
                feature_list.append(lbp.flatten())
            
            # Gabor filter features
            if self.use_gabor:
                from skimage.filters import gabor
                
                # Apply gabor filters at different orientations
                for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                    gabor_real, gabor_imag = gabor(img_norm, frequency=0.6, theta=theta, 
                                                  sigma_x=1.0, sigma_y=1.0)
                    # Use the real part for feature extraction
                    feature_list.append(gabor_real.flatten())
            
            # Advanced edge detection
            if self.use_advanced_edges:
                from skimage.feature import canny
                from skimage.filters import roberts, prewitt
                
                # Canny edge detector (more sophisticated than Sobel)
                canny_edges = canny(img_norm, sigma=1.0)
                feature_list.append(canny_edges.flatten())
                
                # Roberts cross edge detector (good for detailed edges)
                roberts_edges = roberts(img_norm)
                feature_list.append(roberts_edges.flatten())
                
                # Prewitt edge detector (another edge detection variant)
                prewitt_edges = prewitt(img_norm)
                feature_list.append(prewitt_edges.flatten())
            
            # Morphological features
            if self.use_morphology:
                from skimage.morphology import erosion, dilation, opening, closing, disk
                
                # Create structuring element
                selem = disk(1)
                
                # Basic morphological operations
                eroded = erosion(img_norm, selem)
                dilated = dilation(img_norm, selem)
                opened = opening(img_norm, selem)
                closed = closing(img_norm, selem)
                
                # Add morphological features
                feature_list.append(eroded.flatten())
                feature_list.append(dilated.flatten())
                feature_list.append(opened.flatten())
                feature_list.append(closed.flatten())
                
                # Add morphological gradients (difference between operations)
                # External gradient
                ext_gradient = dilated - img_norm
                feature_list.append(ext_gradient.flatten())
                
                # Internal gradient
                int_gradient = img_norm - eroded
                feature_list.append(int_gradient.flatten())
            
            # Histogram of Oriented Gradients (HOG) features
            if self.use_hog:
                from skimage.feature import hog
                
                # Extract HOG features
                # Keep cells and block sizes relatively small for small images
                img_size = img_norm.shape[0]
                pixels_per_cell = max(4, img_size // 8)  # Adapt to image size
                
                hog_features = hog(
                    img_norm, 
                    orientations=8,
                    pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                    cells_per_block=(2, 2),
                    visualize=False,
                    block_norm='L2-Hys'
                )
                feature_list.append(hog_features)
                
        except Exception as e:
            print(f"Warning in feature extraction: {str(e)}")
            # Don't add any features if there's an error


def run_knn_grid_search(X_train, y_train, force_grayscale=False):
    """
    Run grid search for KNN with image preprocessing
    """
    # Define pipeline with preprocessing and KNN
    pipeline = Pipeline([
        ('preprocessor', ImagePreprocessor(force_grayscale=force_grayscale)),
        ('scaler', StandardScaler()),
        ('pca', PCA(random_state=42)),
        ('knn', KNeighborsRegressor())
    ])
    
    # Define parameter grid
    param_grid = {
        'preprocessor__use_edges': [True, False],
        'preprocessor__use_histogram_eq': [True, False],
        'preprocessor__use_clahe': [True, False],
        'preprocessor__use_lbp': [True, False],
        'preprocessor__use_gabor': [True, False],
        'preprocessor__use_advanced_edges': [True, False],
        'preprocessor__use_morphology': [False],  # Morphology adds many features, keeping it off by default
        'preprocessor__use_hog': [True, False],
        'pca__n_components': [100, 150, 200],
        'knn__n_neighbors': [3, 5, 7],
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


def run_focused_knn(X_train, y_train, force_grayscale=False):
    """
    Run KNN with focused parameters (faster than grid search)
    """
    # Create a pipeline with preprocessing
    pipeline = Pipeline([
        ('preprocessor', ImagePreprocessor(
            use_edges=True,
            use_histogram_eq=True,
            use_clahe=True,
            use_lbp=True,
            force_grayscale=force_grayscale,  # Force grayscale mode when needed
            use_gabor=True,              # Enable Gabor filter features
            use_advanced_edges=True,     # Enable advanced edge detection
            use_morphology=False,        # Disable morphology (can add many dimensions)
            use_hog=True                 # Enable HOG features
        )),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=150, random_state=42)),  # Increased components to handle more features
        ('knn', KNeighborsRegressor(n_neighbors=5, weights='distance', p=2))
    ])
    
    # Fit the pipeline
    print("Training focused KNN model...")
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    
    return pipeline


if __name__ == "__main__":
    print("Starting Augmented KNN Pipeline...")
    start_time = time.time()
    
    # Load configs from "config.yaml"
    config = load_config()
    
    # Determine whether to use RGB or grayscale
    USE_RGB = False  # Set to True for RGB, False for grayscale
    
    # Set RGB mode in config
    if USE_RGB:
        if not config.get("load_rgb", False):
            print("[INFO] Setting load_rgb=True for RGB processing")
            config["load_rgb"] = True
    else:
        if config.get("load_rgb", False):
            print("[INFO] Setting load_rgb=False for grayscale processing")
            config["load_rgb"] = False
    
    # Load dataset: images and corresponding minimum distance values
    images, distances, dataset = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")
    
    # Apply data augmentation
    AUGMENT_DATASET = True  # Set to False to skip augmentation
    AUGMENTATION_FACTOR = 3  # How many times to multiply the dataset (e.g., 3 = triple)
    
    if AUGMENT_DATASET:
        augmenter = ImageAugmenter(
            rotation_range=15,       # degrees
            brightness_range=0.3,    # adjust brightness by up to 30%
            flip_horizontal=True,    # horizontal flip
            add_noise=True           # add random noise
        )
        # Augment the dataset
        augmented_images, augmented_distances = augmenter.augment_dataset(
            images, distances, augmentation_factor=AUGMENTATION_FACTOR
        )
    else:
        augmented_images, augmented_distances = images, distances
    
    # For hyperparameter tuning phase: Use train/test split to evaluate performance
    FINAL_SUBMISSION = False  # Set to True for final model submission
    
    if not FINAL_SUBMISSION:
        # Development mode: Use train/test split for validation
        images_train, images_test, distances_train, distances_test = train_test_split(
            augmented_images, augmented_distances, test_size=0.2, random_state=42
        )
        
        # Use either grid search or focused training
        USE_GRID_SEARCH = False  # Set to False for faster focused training
        
        try:
            if USE_GRID_SEARCH:
                # Run grid search to find the best model
                best_model, best_params = run_knn_grid_search(images_train, distances_train)
            else:
                # Run focused training with selected parameters - force grayscale mode when not using RGB
                best_model = run_focused_knn(images_train, distances_train, force_grayscale=not USE_RGB)
            
            # Make predictions on test set
            pred_distances = best_model.predict(images_test)
            
            # Evaluate model
            mae, r2 = print_results(distances_test, pred_distances)
            
            # Visualize results
            #visualize_knn_results(distances_test, pred_distances)
            
        except Exception as e:
            print(f"Error during model training: {str(e)}")
            print("Falling back to basic KNN pipeline...")
            
            # Simple fallback pipeline with minimal preprocessing
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=50, random_state=42)),
                ('knn', KNeighborsRegressor(n_neighbors=5, weights='distance', p=2))
            ])
            
            # Fit and evaluate
            pipeline.fit(images_train, distances_train)
            pred_distances = pipeline.predict(images_test)
            print_results(distances_test, pred_distances)
            best_model = pipeline
    
    else:
        # Final submission mode: Train on ALL available data
        print("\n[INFO]: FINAL SUBMISSION MODE - Using all training data (augmented)\n")
        
        try:
            # Train the model using all available augmented training data
            # Force grayscale mode when not using RGB
            final_model = run_focused_knn(augmented_images, augmented_distances, force_grayscale=not USE_RGB)
            
            # Save the model
            models_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
            os.makedirs(models_folder, exist_ok=True)
            
            augmentation_info = f"_{AUGMENTATION_FACTOR}x" if AUGMENT_DATASET else ""
            rgb_mode = "rgb" if USE_RGB else "gray"
            model_name = f"augmented_knn{augmentation_info}_{rgb_mode}_model.pkl"
            
            
            model_path = os.path.join(models_folder, model_name)
            print(f"Saving final model trained on augmented data to {model_path}...")
            joblib.dump(final_model, model_path)
            print("Final model saved successfully!")
            
            # Load and predict on test data
            try:
                test_images = load_test_dataset(config)
                print(f"[INFO]: Test dataset loaded with {len(test_images)} samples.")
                
                # Make predictions on official test set
                test_predictions = final_model.predict(test_images)
                
                # Save predictions for submission
                save_results(test_predictions)
                print(f"Saved predictions to submission.csv for competition submission")
                
            except Exception as e:
                print(f"Error during test prediction: {str(e)}")
                
        except Exception as e:
            print(f"Error during final model training: {str(e)}")
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")