from flask import Flask, render_template, request
import pickle
from pymongo import MongoClient

app = Flask(__name__)

# Load trained model
with open("student_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load target encoder
with open("target_le.pkl", "rb") as f:
    target_le = pickle.load(f)

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["student_performance_db"]
collection = db["predictions"]

# Home page route
@app.route("/")
def index():
    return render_template("index.html")  # Form page

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect all input fields from the form (5 features)
        features = [
            float(request.form["StudentAbsenceDays"]),
            float(request.form["raisedhands"]),
            float(request.form["VisITedResources"]),
            float(request.form["AnnouncementsView"]),
            float(request.form["Discussion"])
        ]

        # Make prediction
        prediction = model.predict([features])
        result = target_le.inverse_transform(prediction)[0]  # Converts number back to 'L','M','H'

        # Map to readable performance
        performance_map = {"L": "Low", "M": "Medium", "H": "High"}
        outcome = performance_map.get(result, "Unknown")

        # Save input and prediction to MongoDB
        student_record = {
            "StudentAbsenceDays": features[0],
            "RaisedHands": features[1],
            "VisitedResources": features[2],
            "AnnouncementsView": features[3],
            "Discussion": features[4],
            "PredictedPerformance": outcome
        }
        collection.insert_one(student_record)

        return render_template("result.html", prediction=outcome)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
