from utils import load_config, load_dataset, load_test_dataset, print_results, save_results

# sklearn imports:
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate, GridSearchCV # For cross validation
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor  # Models
from sklearn.metrics import mean_squared_error, r2_score  # For predictions analysis, removed from code
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt  # For prediction visualization

# Other imports:
import numpy as np
import os  # For loading previous predictions
import math

# [INFO]: if we don't downscale the images it takes quite long to compute
# SVRs are not allowed in this project.

if __name__ == "__main__":

    number_of_cpu_cores = -1  # Defines the number of the CPU cores to use. Scikit cannot use GPU. -1 for all
    run_pred = False  # Defines if run prediction or not
    run_scores = False  # Define if run scores function or not

    '''1st step: Prepare our data'''

    use_previous_prediction = False  # True/False, load previous predictions if it exists
    previous_pred = "previous_dist_pred.npy"  # Name of previous prediction

    # Load configs from "config.yaml"
    config = load_config()  # Preprocess already implemented in the load_config function

    # Load dataset: images and corresponding minimum distance values
    images, distances, dataset = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")

    scaler = StandardScaler().fit(images)  # StandardScaler implementation
    images_scaled = scaler.transform(images)

    # preprocess_pipeline = Pipeline([
    #     ('scaler', PowerTransformer())
    # ])
    #
    # images = preprocess_pipeline.fit_transform(images)
    # print("Preprocess pipeline applied...")

    processed_images = images_scaled

    # TODO: If you go through the images, you realize that the illumination in
    #  the scene and the objects present in the scene differ a lot between frames. This leads to
    #  variations in pixels values. In order for a machine learning model to perform well, it is
    #  important that the features are scaled in a common range. Therefore, you might need to
    #  use a scaling method from the ones av

    # Small test/debug part:
    # print(distances)  # Distances is a one-dimensional list (NumPy array) of float numbers of size = # of images)
    # print(images)  # Image is a huge list (NumPy array, flattened) containing all RGB or grayscale values (HxLX3 size)
    # desired_image = 0  # Image you want to know the distance of
    # print("The distance for the image {} is: {}".format(desired_image, distances[desired_image]))

    ''' 2nd step: define a model/estimator.
    This needs to be done before splitting the data because the cross validation function requires a model as argument.
    Types of models are: linear regression, tree-based and neural networks (not allowed for this project).
    A big part of having a good code is finding the right model, so we can write several here'''

    ## MODEL 1 ##

    model1 = RandomForestRegressor(random_state=42, n_jobs=number_of_cpu_cores)
    #n_jobs takes the number of CPU cores as argument, -1 means all, here the percentage is defined by the user

    ## MODEL 2 ##

    model2 = GradientBoostingRegressor(
        n_estimators=100,  # Number of boosting rounds (trees)
        learning_rate=0.1,  # Shrinks the contribution of each tree
        max_depth=3,  # Depth of each individual tree
        min_samples_split=2,  # Min samples to split a node
        min_samples_leaf=1,  # Min samples per leaf
        subsample=1.0,  # Use <1.0 for stochastic boosting
        random_state=42  # For reproducibility
    )

    ## MODEL 3 ##

    param_grid = {
        'max_iter': [300, 500, 1000],  # Number of trees, more trees = more power
        'learning_rate': [0.05, 0.1, 0.2],  # Step size, lower = slower learning, more trees needed
        'max_leaf_nodes': [31, 64, 128],  # Bigger trees = more flexibility, but more risk of overfitting
        'min_samples_leaf': [5, 10, 20],  # Regularization: higher = more conservative
        'l2_regularization': [0.0, 1.0, 10.0]  # L2 penalty
    }  # TODO: This is taking tooo much time

    model3 = HistGradientBoostingRegressor(random_state=42)

    grid = GridSearchCV(model3, param_grid, scoring='r2', cv=5, n_jobs=-1, verbose=1)
    grid.fit(processed_images, distances)
    best_model = grid.best_estimator_
    print("Best R²:", grid.best_score_)
    print("Best parameters:", grid.best_params_)

    # Choose model:
    used_model = model3  # Last usage: 3>1>2

    ''' 3rd step: evaluate the dataset.
    Two possible implementations: with scikit.learn train_test_split or cross validation functions.
    
    Cross validation seems to be more robust for this. Scikit has several functions for it, like
    cross_val_score (k-fold cross validation) and cross_validate
    '''

    if run_scores:
        print("Starting score function...")

        # cross_val_score implementation, using a 5-fold cross validation:
        scores = cross_val_score(used_model, processed_images, distances, cv=5, n_jobs=number_of_cpu_cores)

        print("Scores for each fold:", scores)
        print("Mean score:", np.mean(scores))
        print("Standard deviation:", np.std(scores))

        # cross_validate implementation:
        # scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
        #
        # scores = cross_validate(used_model, processed_images, distances, cv=5, scoring=scoring, return_train_score=True)
        #
        # print("R²:", scores['test_r2'].mean())
        # print("MSE:", -scores['test_neg_mean_squared_error'].mean())
        # print("MAE:", -scores['test_neg_mean_absolute_error'].mean())


    ''' 4th step: predict values for all data, if specified to do so
    Comment on cross_val_predict function: For each fold it trains the model on 4/5 of the data predicts on the 1/5
    left out '''

    if run_pred:
        if os.path.exists("previous_dist_pred.npy") & use_previous_prediction:  # Checks if previous prediction exists
            print("Loading cached predictions...")
            dist_pred = np.load("previous_dist_pred.npy")

        else:
            print("Starting prediction function...")
            # Computes the prediction for the distances. For ALL images
            dist_pred = cross_val_predict(used_model, processed_images, distances, cv=5, n_jobs=number_of_cpu_cores)
            # Saving predictions, so that we don't need to always compute again
            np.save("previous_dist_pred.npy", dist_pred)
            print("Finished prediction function...")

        ''' 5th step: visualize the performance
        Important here is to ensure there is no overfitting'''

        print("Starting print results function...")
        print_results(distances, dist_pred)

        # Scatter plot
        plt.figure(figsize=(6, 6))
        plt.scatter(distances, dist_pred, alpha=0.5)
        plt.plot([distances.min(), distances.max()], [distances.min(), distances.max()], 'r--')  # Perfect prediction line
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title("Predicted vs Actual Values")
        plt.grid(True)
        plt.tight_layout()
        # plt.show()
        plt.savefig("pred_vs_true.png")

    # Save the results
    # save_results(test_pred)