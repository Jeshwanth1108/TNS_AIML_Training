import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Dataset path

DATA_PATH = "../dataset/Heart_Disease_Prediction.csv"


# Load dataset

df = pd.read_csv(DATA_PATH)


print("Dataset Loaded")
print(df.head())


# Convert target

encoder = LabelEncoder()

df["Heart Disease"] = encoder.fit_transform(
    df["Heart Disease"]
)


print(
    df["Heart Disease"].value_counts()
)


# Features and target

X = df.drop(
    "Heart Disease",
    axis=1
)

y = df["Heart Disease"]



# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



# Scaling

scaler = StandardScaler()


X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)



# Model

model = LogisticRegression()


model.fit(
    X_train,
    y_train
)



# Evaluation

prediction = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    prediction
)


print(
    "Accuracy:",
    accuracy
)


print(
    classification_report(
        y_test,
        prediction
    )
)



# Save model

os.makedirs(
    "models",
    exist_ok=True
)


joblib.dump(
    model,
    "models/heart_model.pkl"
)


joblib.dump(
    scaler,
    "models/scaler.pkl"
)


joblib.dump(
    encoder,
    "models/encoder.pkl"
)


print(
    "Model saved successfully"
)