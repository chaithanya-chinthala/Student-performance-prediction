import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("dataset/xAPI-Edu-Data.csv")

# Encode categorical columns (except target)
le_features = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object' and col != 'Class':
        data[col] = le_features.fit_transform(data[col])

# Features (5 features)
X = data[['StudentAbsenceDays', 'raisedhands', 'VisITedResources', 
          'AnnouncementsView', 'Discussion']]

# Encode target (Class: L,M,H)
target_le = LabelEncoder()
y = target_le.fit_transform(data['Class'])  # Converts L,M,H → 0,1,2

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test the model
test_input = [[20, 30, 40, 10, 15]]  # Example input for 5 features
pred = model.predict(test_input)
pred_class = target_le.inverse_transform(pred)[0]  # Convert 0/1/2 → L/M/H
print("Test prediction:", pred_class)

# Save trained model and target encoder
with open("student_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("target_le.pkl", "wb") as f:
    pickle.dump(target_le, f)

print("✅ Model and target encoder saved successfully!")
