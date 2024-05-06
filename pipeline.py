from utils import (
    load_config,
    load_dataset,
    load_private_test_dataset,
    print_results,
    save_results,
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


class SobelEdgeDetector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([sobel(image) for image in X])


if __name__ == "__main__":
    config = load_config()

    images, distances = load_dataset(config, "train")
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")

    train_images, test_images, train_distances, test_distances = train_test_split(
        images, distances, test_size=0.1, random_state=42
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
        # "pca__n_components": [30],  # Adjusted to maximum allowed [20, 30, 40]
        "rf__n_estimators": [500],
        "rf__max_depth": [30, 60],
        "rf__min_samples_split": [2],
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
        verbose=2,
    )

    grid_search.fit(train_images, train_distances)
    print(f"[INFO]: Best Model Parameters: {grid_search.best_params_}")

    datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model = grid_search.best_estimator_
    model_params_str = "dont_care"
    # f"n_components_{best_model.named_steps['pca'].n_components}_n_estimators_{best_model.named_steps['rf'].n_estimators}"
    model_name = f"{datetime_str}_RandomForest_model_downsample_{model_params_str}.pkl"

    pred_distances = best_model.predict(test_images)
    print_results(test_distances, pred_distances)

    models_folder = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_folder, exist_ok=True)

    model_path = os.path.join(models_folder, model_name)
    joblib.dump(best_model, model_path)