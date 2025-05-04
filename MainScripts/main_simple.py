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
    # ('model', RandomForestRegressor(random_state=42, n_jobs=-1))  # long, good results
    ('model', HistGradientBoostingRegressor(random_state=42))  # fast, good results without pca
])

"Fitting the data and exporting the pipeline"

pipeline.fit(images_train, distances_train)

# # Save the entire pipeline
# joblib.dump(pipeline, 'model_pipeline.pkl')

"Evaluate the model"

print_results(distances_test, pipeline.predict(images_test))


