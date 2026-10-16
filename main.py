import joblib

try:
    # Tell Python to reach inside and load the binary file
    model = joblib.load("best_bike_sharing_model.joblib")
    print("Success! Your model file loaded into memory with zero errors.")
    print("The model object is:", type(model))
except Exception as e:
    print(f"Error loading file: {e}")
