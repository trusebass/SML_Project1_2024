from utils import (
    load_config,
    load_dataset,
    load_test_dataset,
    print_results,
    save_results,
    log_model_results,
)
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from skimage.filters import sobel
from skimage.feature import graycomatrix, graycoprops
from sklearn.base import BaseEstimator, TransformerMixin
import os
import datetime
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
import numpy as np
import time
import matplotlib.pyplot as plt
import sys

# Detect if running in Google Colab
def is_running_in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

IS_COLAB = is_running_in_colab()


class SobelEdgeDetector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Add progress bar for the edge detection transformation
        if IS_COLAB:
            print("Applying Sobel edge detection...")
            result = []
            for i, image in enumerate(X):
                if i % 10 == 0:
                    print(f"Processing image {i+1}/{len(X)}")
                result.append(sobel(image))
            print("Sobel edge detection completed")
            return np.array(result)
        else:
            return np.array([sobel(image) for image in tqdm(X, desc="Applying Sobel edge detection")])


class TextureFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract GLCM texture features for improved image analysis"""
    def __init__(self, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        self.distances = distances
        self.angles = angles
        self.props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # Reshape images for GLCM
        n_samples = X.shape[0]
        image_size = int(np.sqrt(X.shape[1]))
        
        print("Extracting texture features...")
        features = []
        
        for i, img in enumerate(tqdm(X, desc="Extracting GLCM features")):
            # Reshape to 2D
            img_2d = img.reshape(image_size, image_size)
            
            # Normalize to [0, 255] and convert to uint8 for GLCM
            img_norm = (img_2d - img_2d.min()) * 255.0 / (img_2d.max() - img_2d.min())
            img_uint8 = img_norm.astype(np.uint8)
            
            # Calculate GLCM
            glcm = graycomatrix(img_uint8, self.distances, self.angles, 256, symmetric=True, normed=True)
            
            # Extract properties
            feature = []
            for prop in self.props:
                feature.extend(graycoprops(glcm, prop).flatten())
            
            features.append(feature)
            
        return np.array(features)


class FeatureAnalyzer(BaseEstimator, TransformerMixin):
    """Transforms data while storing feature importance information"""
    def __init__(self):
        self.feature_importances_ = None
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return X
        
    def set_feature_importances(self, importances):
        self.feature_importances_ = importances


class AdvancedPipelineBuilder:
    """Helper class to build pipeline with enhanced features"""
    def __init__(self, config, train_images, train_distances):
        self.config = config
        self.train_images = train_images
        self.train_distances = train_distances
        self.feature_analyzer = FeatureAnalyzer()
        
    def visualize_pca_variance(self, X, n_components=None):
        """Analyze and visualize the variance explained by PCA components"""
        # Apply scaler first
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit PCA for analysis
        pca = PCA().fit(X_scaled)
        
        # Calculate cumulative variance
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        
        # Plot variance explained
        plt.figure(figsize=(10, 5))
        plt.plot(cumulative_variance, 'b-', linewidth=2)
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance')
        plt.grid(True)
        
        # Add 95% line
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
        
        # Find number of components for 95% variance
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        plt.axvline(x=n_components_95, color='g', linestyle='--', 
                   label=f'Components for 95% variance: {n_components_95}')
        
        plt.legend()
        
        # Save the plot
        plot_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, "pca_variance.png"))
        
        if IS_COLAB:
            plt.show()
        else:
            plt.close()
            
        print(f"PCA analysis: {n_components_95} components needed for 95% variance")
        return n_components_95 if n_components is None else n_components
    
    def build_pipeline(self, use_texture=True, use_stacking=False, pca_components=None):
        """Build an enhanced pipeline with advanced features"""
        steps = []
        
        # Start with preprocessing
        if use_texture:
            steps.append(("texture", TextureFeatureExtractor(
                distances=[1],  # Use only immediate neighbors
                angles=[0, np.pi/2]  # Use only horizontal and vertical directions
            )))
        
        # Add edge detection if needed
        steps.append(("edge_detection", SobelEdgeDetector()))
        
        # Add feature analyzer for later inspection
        steps.append(("analyzer", self.feature_analyzer))
        
        # Add scaling
        steps.append(("scaler", RobustScaler()))
        
        # Determine number of PCA components if not specified
        if pca_components is None:
            # Process images directly for PCA analysis
            # We'll use all the preprocessing steps except create a copy for the PCA analysis
            # without duplicating step names
            processed_images = self.train_images
            
            # Apply each preprocessing step individually to avoid pipeline name conflicts
            for step_name, transformer in steps:
                processed_images = transformer.fit_transform(processed_images)
            
            # Visualize PCA and get optimal components
            pca_components = self.visualize_pca_variance(processed_images)
        
        # Add PCA with determined components
        steps.append(("pca", PCA(n_components=pca_components)))
        
        # Add final estimator (RandomForest or Stacked model)
        if use_stacking:
            # Create a stacked model with RF, Ridge, and KNN
            estimators = [
                ('rf', RandomForestRegressor(n_jobs=-1, random_state=42)),
                ('ridge', Ridge(random_state=42)),
                ('knn', KNeighborsRegressor(n_jobs=-1))
            ]
            steps.append(("stacked", VotingRegressor(estimators)))
        else:
            steps.append(("rf", RandomForestRegressor(
                n_jobs=-1, 
                random_state=42,
                warm_start=True,
                max_samples=0.8  # Use 80% of samples for each tree, speeds up training
            )))
        
        return Pipeline(steps)


class ColabFriendlyGridSearchCV(GridSearchCV):
    """Custom GridSearchCV that provides progress updates suitable for Colab"""
    def fit(self, X, y=None, **fit_params):
        # Import here to avoid issues
        from sklearn.model_selection import ParameterGrid

        # Get parameters for tracking progress
        n_splits = self.cv
        param_iterator = list(ParameterGrid(self.param_grid))
        n_candidates = len(param_iterator)
        n_fits = n_splits * n_candidates
        
        print(f"Running {n_fits} fits for {n_candidates} candidates with {n_splits}-fold cross-validation")
        
        # Set up progress tracking
        self.start_time = time.time()
        self.n_fits_completed = 0
        self.total_fits = n_fits
        self.last_print_time = self.start_time
        self.print_interval = 10  # seconds between progress updates
        
        # Store original verbose setting and set to 0
        original_verbose = self.verbose
        self.verbose = 0
        
        # Run the actual fitting
        result = super().fit(X, y, **fit_params)
        
        # Calculate and report training time
        self.training_time = time.time() - self.start_time
        print(f"GridSearchCV completed in {self.training_time:.2f}s")
        
        return result
    
    def _fit_and_score(self, estimator, X, y, *args, **kwargs):
        # Run the original fit_and_score
        result = super()._fit_and_score(estimator, X, y, *args, **kwargs)
        
        # Update progress counter
        self.n_fits_completed += 1
        
        # Print progress at intervals
        current_time = time.time()
        if current_time - self.last_print_time > self.print_interval:
            elapsed = current_time - self.start_time
            progress = (self.n_fits_completed / self.total_fits) * 100
            
            # Estimate remaining time
            if self.n_fits_completed > 0:
                avg_time_per_fit = elapsed / self.n_fits_completed
                remaining_fits = self.total_fits - self.n_fits_completed
                estimated_remaining = avg_time_per_fit * remaining_fits
                
                # Format as HH:MM:SS
                remaining_str = str(datetime.timedelta(seconds=int(estimated_remaining)))
                elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
                
                print(f"Progress: {self.n_fits_completed}/{self.total_fits} fits ({progress:.1f}%) - "
                      f"Elapsed: {elapsed_str} - Estimated remaining: {remaining_str}")
            else:
                print(f"Progress: {self.n_fits_completed}/{self.total_fits} fits ({progress:.1f}%)")
                
            self.last_print_time = current_time
            
            # Force output to display immediately in Colab
            sys.stdout.flush()
            
        return result


if __name__ == "__main__":
    config = load_config()

    images, distances, dataset = load_dataset(config, "train")
    print(f"[INFO]: Dataset {dataset} loaded with {len(images)} samples.")

    train_images, test_images, train_distances, test_distances = train_test_split(
        images, distances, test_size=0.2, random_state=42
    )

    # Build enhanced pipeline with advanced features
    pipeline_builder = AdvancedPipelineBuilder(config, train_images, train_distances)
    
    # You can customize these settings
    use_texture = True  # Use GLCM texture features
    use_stacking = True  # Use stacked model (RF + Ridge + KNN)
    
    # Build the pipeline (PCA components will be determined automatically)
    pipeline = pipeline_builder.build_pipeline(
        use_texture=use_texture,
        use_stacking=use_stacking
    )

    # Extended parameter grid for better tuning
    if use_stacking:
        param_grid = {
            'stacked__rf__n_estimators': [100, 500],
            'stacked__rf__max_depth': [30, None],
            'stacked__rf__min_samples_leaf': [1, 4],
            'stacked__ridge__alpha': [0.1, 10.0],
            'stacked__knn__n_neighbors': [3, 7]
        }
    else:
        param_grid = {
            'rf__n_estimators': [100, 500],  # Reduced from [100, 500, 1000]
            'rf__max_depth': [30, None],     # Reduced from [30, 60, None]
            'rf__min_samples_leaf': [1, 4]   # Reduced from [1, 2, 4]
        }

    # Choose the appropriate GridSearchCV implementation based on environment
    if IS_COLAB:
        print("[INFO]: Running in Google Colab environment")
        grid_search = ColabFriendlyGridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring="neg_mean_absolute_error",
            verbose=0,
        )
    else:
        # Use normal GridSearchCV with verbose=2 for non-Colab environments
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring="neg_mean_absolute_error",
            verbose=2,
        )
        # For tracking training time
        start_time = time.time()

    print("[INFO]: Starting grid search...")
    # Training time is tracked inside the GridSearchCV class
    grid_search.fit(train_images, train_distances)
    
    # For non-Colab environments, calculate training time
    if not IS_COLAB:
        training_time = time.time() - start_time
    else:
        training_time = grid_search.training_time
    
    print(f"[INFO]: Best Model Parameters: {grid_search.best_params_}")

    datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model = grid_search.best_estimator_
    
    # Extract more meaningful model name
    if use_stacking:
        model_type = "StackedModel"
    else:
        model_type = "RandomForest"
        
    texture_str = "with_texture" if use_texture else "no_texture"
    model_name = f"{datetime_str}_{model_type}_{texture_str}.pkl"

    # Add progress bar for predictions
    print("Generating predictions...")
    pred_distances = best_model.predict(test_images)
    
    # Print results and get metrics
    mae, r2 = print_results(test_distances, pred_distances)

    # Feature importance analysis
    feature_analyzer = best_model.named_steps.get('analyzer')
    if not use_stacking and 'rf' in best_model.named_steps:
        # Store feature importances in the analyzer
        feature_analyzer.set_feature_importances(best_model.named_steps['rf'].feature_importances_)
        
        # Plot feature importance
        feat_importances = best_model.named_steps['rf'].feature_importances_
        n_features = min(20, len(feat_importances))  # Show top 20 features
        indices = np.argsort(feat_importances)[-n_features:]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(n_features), feat_importances[indices])
        plt.yticks(range(n_features), [f"Feature {i}" for i in indices])
        plt.xlabel("Feature Importance")
        plt.title("Top Feature Importances")
        
        # Save feature importance plot
        plot_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, "feature_importance.png"))
        
        if IS_COLAB:
            plt.show()
        else:
            plt.close()

    # Log model results with training time
    best_params = grid_search.best_params_
    
    # Create additional info dict with all the enhancements used
    additional_info = {
        "model_type": "Stacked Pipeline" if use_stacking else "Pipeline with RandomForest",
        "dataset_size": len(images),
        "train_size": len(train_images),
        "test_size": len(test_images),
        "grid_search_best_score": grid_search.best_score_,
        "pipeline_steps": [step[0] for step in pipeline.steps],
        "used_texture_features": use_texture,
        "used_stacking": use_stacking,
        "config_rgb": config["load_rgb"],
        "config_downsample": config["downsample_factor"]
    }
    
    log_model_results(
        model_name=model_name,
        model_params=best_params,
        mae=mae,
        r2=r2,
        training_time=training_time,
        additional_info=additional_info
    )

    models_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_folder, exist_ok=True)

    model_path = os.path.join(models_folder, model_name)
    print(f"Saving model to {model_path}...")
    joblib.dump(best_model, model_path)
    print("Model saved successfully!")