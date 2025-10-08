from flask import Flask, render_template, request
import pickle
from pymongo import MongoClient
from datetime import datetime
import matplotlib.pyplot as plt

app = Flask(__name__)

# ------------------ LOAD MODEL ------------------ #
with open("student_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("target_le.pkl", "rb") as f:
    target_le = pickle.load(f)


# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["student_performance_db"]
collection = db["predictions"]



performance_map = {"L": "Low", "M": "Medium", "H": "High"}

# ------------------ ROUTES ------------------ #

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Collect input features in same order as model training
    # Collect form input
    features = [
        float(request.form["StudentAbsenceDays"]),
        float(request.form["raisedhands"]),
        float(request.form["VisITedResources"]),
        float(request.form["AnnouncementsView"]),
        float(request.form["Discussion"])
    ]

    # Make prediction using trained model
    prediction = model.predict([features])

    # Convert numeric prediction back to L/M/H
    result = target_le.inverse_transform(prediction)[0]

    # Map to readable performance
    performance_map = {"L": "Low", "M": "Medium", "H": "High"}
    outcome = performance_map.get(result, "Unknown")


    # Save to MongoDB with native Python types
    student_id = str(request.form.get("StudentID", "1"))
    record = {
        "StudentID": student_id,
        "StudentAbsenceDays": float(features[0]),
        "RaisedHands": float(features[1]),
        "VisitedResources": float(features[2]),
        "AnnouncementsView": float(features[3]),
        "Discussion": float(features[4]),
        "PredictedPerformance": str(outcome),  # store as string
        "timestamp": datetime.now()
    }
    collection.insert_one(record)

    return render_template("result.html", prediction=outcome, student_id=student_id)


@app.route("/progress/<student_id>")
def progress(student_id):
    records = list(collection.find({"StudentID": student_id}).sort("timestamp"))
    if not records:
        return f"No records found for Student ID {student_id}"

    # Prepare data for plotting
    dates = [r["timestamp"].strftime("%Y-%m-%d %H:%M") for r in records]
    mapping = {"Low": 1, "Medium": 2, "High": 3}
    performance_numeric = [int(mapping.get(r["PredictedPerformance"], 0)) for r in records]

    # Plot progress
    plt.figure(figsize=(6,4))
    plt.plot(dates, performance_numeric, marker='o', linestyle='-', color="#3498db")
    plt.title(f"Performance Progress for Student {student_id}")
    plt.ylabel("Performance (1-L, 2-M, 3-H)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    chart_path = f"static/progress_{student_id}.png"
    plt.savefig(chart_path)
    plt.close()

    return render_template("progress.html", student_id=student_id, chart_path=chart_path)


@app.route("/dashboard")
def dashboard():
    data = list(collection.find())
    counts = {"Low": 0, "Medium": 0, "High": 0}
    for r in data:
        perf = r.get("PredictedPerformance", "Low")  # default to Low if missing
        if perf in counts:
            counts[perf] += 1

    # Pie chart of class performance
    plt.figure(figsize=(5,5))
    plt.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', startangle=90,
            colors=["#e74c3c","#f1c40f","#2ecc71"])
    plt.title("Class Performance Distribution")
    chart_path = "static/performance_chart.png"
    plt.savefig(chart_path)
    plt.close()

    return render_template("dashboard.html", chart_path=chart_path)


if __name__ == "__main__":
    app.run(debug=True)

