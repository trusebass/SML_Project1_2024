from utils import load_config, load_dataset, load_private_test_dataset, print_results, save_results
import joblib
import yaml
from sklearn.model_selection import train_test_split


# Load configs from "config.yaml"
config = load_config()

# Load the trained model
model = joblib.load('models/RandomForestRegressor()_5_T.pkl')

# Load your new test set from a different folder (not test)
#'''
new_images, new_distances, dataset = load_dataset(config,"public_test")
print(f"[INFO]: Dataset {dataset} loaded with {len(new_images)} samples.")
#'''

#To test on Data from the Training Set:
'''
training_images, training_distances = load_dataset(config,"train")
train_images, new_images, train_distances, new_distances = train_test_split(training_images, training_distances, test_size=0.2, random_state=42)
print(f"[INFO]: Dataset loaded with {len(new_images)} samples.")
'''

# Make predictions on the new test set
pred_distances = model.predict(new_images)

# Print and save the new results
print_results(new_distances, pred_distances)
#save_results(new_predictions)