from utils import load_config, load_dataset, load_private_test_dataset, print_results, save_results
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os
import joblib

# sklearn imports...

if __name__ == "__main__":
    # Load configs from "config.yaml"
    config = load_config()

    # Load dataset: images and corresponding minimum distance values
    images, distances, dataset = load_dataset(config,"train")
    print(f"[INFO]: Dataset {dataset} loaded with {len(images)} samples.")

    
    
    # TODO: Your implementation starts here

    # split dataset into training and testing
    train_images, test_images, train_distances, test_distances = train_test_split(images, distances, test_size=0.2, random_state=42)

    
    # Create the Random Forests model
    model = RandomForestRegressor(n_jobs = -1, random_state = 42)

    # Train the model
    model.fit(train_images, train_distances)



    ## name the trained model
    model_name = f"{str(model)}_model.pkl"
    
    ## Create the models folder if it doesn'aölkdsfjt exist
    models_folder = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_folder, exist_ok=True)

    # Save the trained model
    model_path = os.path.join(models_folder, model_name)
    joblib.dump(model, model_path)



    # Make predictions
    pred_distances = model.predict(test_images)


    # Print the results
    print_results(test_distances, pred_distances)
    

    # Save the resultsts
    #save_results(predictions)