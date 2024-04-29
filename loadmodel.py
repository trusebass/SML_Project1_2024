from utils import load_config, load_dataset, load_private_test_dataset, print_results, save_results
from sklearn.externals import joblib

# Load the trained model
model = joblib.load('models/trained_model.pkl')

# Load your new test set
new_images, new_distances = load_dataset(config)
print(f"[INFO]: Dataset loaded with {len(new_images)} samples.")

# Make predictions on the new test set
new_predictions = model.predict(new_images)

# Print and save the new results
print_results(new_distances, new_predictions)
save_results(new_predictions)