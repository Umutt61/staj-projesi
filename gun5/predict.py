import sys
import joblib

if len(sys.argv) < 2:
    print("Lütfen mesaj girin: python predict.py \"mesaj\"")
    sys.exit(1)

text = sys.argv[1]

model = joblib.load("best_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

vec = vectorizer.transform([text])
prediction = model.predict(vec)[0]

print(f"✅ Tahmin: {prediction}")