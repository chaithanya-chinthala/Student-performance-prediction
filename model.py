import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("dataset/xAPI-Edu-Data.csv")

# Encode categorical features (keep target separate)
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object' and col != 'Class':
        data[col] = le.fit_transform(data[col])

# Features and target
X = data[['StudentAbsenceDays', 'raisedhands', 'VisITedResources', 
          'AnnouncementsView', 'Discussion']]

# Encode target
target_le = LabelEncoder()
y = target_le.fit_transform(data['Class'])  # 0,1,2

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and target encoder
with open("student_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("target_le.pkl", "wb") as f:
    pickle.dump(target_le, f)

print("✅ Model and target encoder saved successfully")
