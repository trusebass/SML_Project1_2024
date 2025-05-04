from utils import load_config, load_dataset, load_test_dataset, print_results, save_results

# Other imports:
import numpy as np
import joblib
import math
import matplotlib.pyplot as plt  # For prediction visualization
# sklearn imports:
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, cross_validate, GridSearchCV # For cross validation
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor  # Models

from sklearn.metrics import mean_squared_error, r2_score  # For predictions analysis, removed from code

from sklearn.neighbors import KNeighborsRegressor


"Load the data"
# Load configs from "config.yaml"
config = load_config()

# Load dataset: images and corresponding minimum distance values
train_images, distances, dataset = load_dataset(config)
print(f"[INFO]: Dataset loaded with {len(train_images)} samples.")

# images_test = load_test_dataset(config)
# print(f"[INFO]: Test dataset loaded with {len(images_test)} samples.")

images_train, images_test, distances_train, distances_test = train_test_split(train_images, distances, test_size=0.2, random_state=42)

"Pipeline definition"

# Define pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=5)),
    ('model', KNeighborsRegressor())
])

"GridSearch for optimal parameters"

# Define parameter grid
param_grid = {
    'model__n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'model__weights': ['uniform', 'distance'],
    'model__p': [1, 2],  # p=1 for Manhattan, p=2 for Euclidean
    'model__leaf_size': [20, 30, 40, 50],
    'pca__n_components': [3, 5, 7, 10, 15]
}

# Create grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)

"Fitting the data and finding the best parameters"

# Fit the grid search
grid_search.fit(images_train, distances_train)

# Print best parameters
print("Best parameters:", grid_search.best_params_)
print("Best score:", -grid_search.best_score_)  # Convert back to positive MSE

# Use the best estimator for predictions
best_pipeline = grid_search.best_estimator_

"Evaluate the model"

print_results(distances_test, best_pipeline.predict(images_test))

# Save the best pipeline if needed
joblib.dump(best_pipeline, 'best_knn_pipeline.pkl')