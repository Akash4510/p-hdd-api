from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the mmodel
model = joblib.load("models/logistic_regression_model.joblib")

# Load your scaler (saved using joblib)
scaler = joblib.load('models/scaler.pkl')


@app.get("/")
def home():
    return jsonify({
        "message": "Server running..."
    })


feature_names = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                 "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]


@app.route('/predict', methods=['POST'])
def predict():
    # Extract JSON data from the request body
    data = request.get_json()

    # Check if the required keys are present in the data
    if not all(key in data for key in feature_names):
        return jsonify({"error": "Invalid input, some fields are missing"}), 400

    # Create a list of all values
    values_list = [data[key] for key in feature_names]

    # Create a DataFrame with the input data
    input_df = pd.DataFrame([values_list], columns=feature_names)

    print(input_df)

    # Scale input data using the loaded scaler
    scaled_data = scaler.transform(input_df)

    print(scaled_data)

    # Predict using the model
    prediction = model.predict(scaled_data)

    prediction_message = "Your heart seems to be in a healthy condition" if prediction[
        0] == 0 else "Your heart seems to be unhealthy, please consult your doctor"

    return jsonify({
        "message": "Prediction successful",
        "prediction": int(prediction[0]),
        "prediction_message": prediction_message
    })


if __name__ == "__main__":
    app.run(debug=True)
