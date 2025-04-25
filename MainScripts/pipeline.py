from utils import (
    load_config,
    load_dataset,
    load_test_dataset,
    print_results,
    save_results,
    log_model_results,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from skimage.filters import sobel
from sklearn.base import BaseEstimator, TransformerMixin
import os
import datetime
import joblib
from sklearn.preprocessing import RobustScaler
import numpy as np
import time


class SobelEdgeDetector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Add progress bar for the edge detection transformation
        return np.array([sobel(image) for image in tqdm(X, desc="Applying Sobel edge detection")])


class TqdmGridSearchCV(GridSearchCV):
    """Custom GridSearchCV that shows a progress bar using tqdm"""
    def fit(self, X, y=None, **fit_params):
        from sklearn.model_selection import ParameterGrid
        n_splits = self.cv
        param_iterator = list(ParameterGrid(self.param_grid))
        n_candidates = len(param_iterator)
        n_fits = n_splits * n_candidates
        
        print(f"Running {n_fits} fits for {n_candidates} candidates with {n_splits}-fold cross-validation")
        self.verbose = 0  # Disable built-in verbosity
        
        self.pbar = tqdm(total=n_fits, desc="GridSearchCV")
        self.start_time = time.time()  # Track start time
        result = super().fit(X, y, **fit_params)
        self.training_time = time.time() - self.start_time  # Calculate training time
        
        # Update progress bar to completion in case we couldn't track individual steps
        self.pbar.update(n_fits - self.pbar.n)
        self.pbar.set_description(f"GridSearchCV completed in {self.training_time:.2f}s")
        self.pbar.close()
        
        return result
    
    def _fit_and_score(self, estimator, X, y, *args, **kwargs):
        result = super()._fit_and_score(estimator, X, y, *args, **kwargs)
        self.pbar.update(1)
        return result


if __name__ == "__main__":
    config = load_config()

    images, distances, dataset = load_dataset(config, "train")
    print(f"[INFO]: Dataset {dataset} loaded with {len(images)} samples.")

    train_images, test_images, train_distances, test_distances = train_test_split(
        images, distances, test_size=0.2, random_state=42
    )

    pipeline = Pipeline(
        [
            # ("edge_detection", SobelEdgeDetector()),
            ("scaler", RobustScaler()),
            (
                "pca",
                PCA(n_components=min(99, len(train_images[0]))),
            ),  # adjust based on actual feature or sample size, reduces dimensions to 49 or train_images length, whatever is shorter
            ("rf", RandomForestRegressor(n_jobs=-1)),
        ]
    )

    param_grid = {
        "pca__n_components": [30],  # Adjusted to maximum allowed [20, 30, 40]
        "rf__n_estimators": [500],
        "rf__max_depth": [30, 60],
        "rf__min_samples_split": [2],
    }

    # Use our custom GridSearchCV with tqdm progress bar
    #grid_search = TqdmGridSearchCV(
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
        verbose=0,  # Set to 0 since we're using our own progress tracking
    )

    print("[INFO]: Starting grid search...")
    # Training time is tracked inside the TqdmGridSearchCV class
    grid_search.fit(train_images, train_distances)
    training_time = grid_search.training_time
    
    print(f"[INFO]: Best Model Parameters: {grid_search.best_params_}")

    datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model = grid_search.best_estimator_
    model_params_str = "dont_care"
    # f"n_components_{best_model.named_steps['pca'].n_components}_n_estimators_{best_model.named_steps['rf'].n_estimators}"
    model_name = f"{datetime_str}_RandomForest_model_downsample_{model_params_str}.pkl"

    # Add progress bar for predictions
    print("Generating predictions...")
    pred_distances = best_model.predict(test_images)
    
    # Print results and get metrics
    mae, r2 = print_results(test_distances, pred_distances)

    # Log model results with training time
    best_params = grid_search.best_params_
    log_model_results(
        model_name=model_name,
        rgb = config["load_rgb"],
        downsample = config["downsample_factor"],
        model_params=best_params,
        mae=mae,
        r2=r2,
        training_time=training_time,
        additional_info={
            "model_type": "Pipeline with RandomForest",
            "dataset_size": len(images),
            "train_size": len(train_images),
            "test_size": len(test_images),
            "grid_search_best_score": grid_search.best_score_,
            "pipeline_steps": [step[0] for step in pipeline.steps]
        }
    )

    models_folder = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_folder, exist_ok=True)

    # model_path = os.path.join(models_folder, model_name)
    # print(f"Saving model to {model_path}...")
    # joblib.dump(best_model, model_path)
    # print("Model saved successfully!")