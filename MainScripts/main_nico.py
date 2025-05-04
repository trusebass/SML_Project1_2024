from utils import load_config, load_dataset, load_test_dataset, print_results, save_results

# sklearn imports:
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, cross_validate, GridSearchCV # For cross validation
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, PowerTransformer, FunctionTransformer
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor  # Models
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge

from skimage.transform import resize
from skimage.filters import unsharp_mask
from skimage.exposure import rescale_intensity, equalize_hist, equalize_adapthist
from skimage.restoration import denoise_tv_chambolle

from skimage.filters import sobel
from skimage.feature import canny
from skimage.transform import resize

# Other imports:
import numpy as np

"Load and split the train data"

# Load configs from "config.yaml"
config = load_config()

# Load dataset: images and corresponding minimum distance values
train_images, distances = load_dataset(config)  # mod: non flattened images
print(f"[INFO]: Dataset loaded with {len(train_images)} samples.")

# # Split the data
# images_train, images_test, distances_train, distances_test = (
#     train_test_split(train_images, distances, test_size=0.2, random_state=42))

"Preprocess function and pipeline definition"

# Preprocess function
def preprocess_images(images):
    processed = []
    for img in images:

        # img = equalize_adapthist(img, clip_limit=0.03)  # adaptive/local exposure equalization
        # img = equalize_hist(img)  # global exposure histogram equalization
        # img = denoise_tv_chambolle(img, weight=0.1)  # denoise
        # img = rescale_intensity(img) # contrast stretch
        # img = resize(img, (30,30), anti_aliasing=False)  # down sampling, not as they do (this is bad)

        processed.append(img.flatten())  # flattens the image to a 1D array
    return np.array(processed)

# Define pipeline
pipeline = Pipeline([

    ('preprocess', FunctionTransformer(preprocess_images)),  # mod: flatten necessary
    ('scaler', StandardScaler()),
    ('pca', PCA(random_state=42)),  # Use only with downscale factor 1

    # ('model', HistGradientBoostingRegressor(random_state=42))  # fast, good results, no PCA and scaler
    # ('model', ExtraTreesRegressor(n_estimators=300, n_jobs=-1, random_state=42))  # no PCA, with scaler, no RGB
    ('model', KNeighborsRegressor())  # fast, needs scaler, no PCA

])


"Fit the data and export the pipeline (optional)"

param_grid = {
    'pca__n_components': [20, 30, 40, 50, 100, 150, 200],
    'model__n_neighbors': [1, 2, 3, 4, 5, 7, 11],
    'model__weights': ['distance', 'uniform'],
    'model__p': [1, 2]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)

grid.fit(train_images, distances)
best_estimator = grid.best_estimator_

print("Best params:", grid.best_params_)
print("Best MAE:", -grid.best_score_*100)

# # Save the entire pipeline
# joblib.dump(pipeline, 'model_pipeline.pkl')

"Load test_images, predict with best estimator and save"

test_images = load_test_dataset(config)
print(f"[INFO]: Test dataset loaded with {len(test_images)} samples.")

save_results(best_estimator.predict(test_images))
print(f"[INFO]: Predictions saved successfully")
