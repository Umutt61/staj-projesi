import joblib
import json

model = joblib.load("best_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

texts = [
    "You won a free iPhone! Click this link.",
    "Hey, are we still on for tonight?",
    "Congratulations! You have been selected for a prize.",
    "Let's catch up tomorrow after work.",
    "Claim your free vacation now!"
]

results = {text: model.predict(vectorizer.transform([text]))[0] for text in texts}

with open("pred_samples.json", "w") as f:
    json.dump(results, f, indent=2)

print("✅ 5 örnek mesaj tahmini kaydedildi.")