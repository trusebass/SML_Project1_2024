from utils import load_config, load_dataset, load_private_test_dataset, print_results, save_results
from sklearn.ensemble import RandomForestRegressor

# sklearn imports...

if __name__ == "__main__":
    # Load configs from "config.yaml"
    config = load_config()

    # Load dataset: images and corresponding minimum distance values
    images, distances = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")

    
    
    # TODO: Your implementation starts here
    
    
    # Create the Random Forests model
    model = RandomForestRegressor()

    # Train the model
    model.fit(images, distances)

    # Make predictions
    predictions = model.predict(images)

    # TODO: Add any additional preprocessing steps, training, and evaluation code here

    # Print the results
    print_results(distances, predictions)

    # Save the resultsts
    save_results(predictions)