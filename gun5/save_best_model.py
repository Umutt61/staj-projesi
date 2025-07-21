import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

mlflow.set_tracking_uri("file:../mlruns")
client = MlflowClient()

with open("C:/Users/umut/staj-projesi/gun4/best_run.txt") as f:
    best_run_id = f.read().strip()

run_data = client.get_run(best_run_id)
model_name = run_data.data.params["model_name"]

df = pd.read_csv("../gun2/sms_clean.csv")
X_train, _, y_train, _ = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

if model_name == "LogisticRegression":
    model = LogisticRegression(max_iter=1000)
elif model_name == "MultinomialNB":
    model = MultinomialNB()
elif model_name == "RandomForest":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    raise ValueError("Model bilinmiyor.")

model.fit(X_train_vec, y_train)

joblib.dump(model, "best_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✅ Model ve vectorizer kaydedildi.")