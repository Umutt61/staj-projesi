from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Model ve vectorizer yükle
model = joblib.load(os.path.join("..", "gun5", "best_model.pkl"))
vectorizer = joblib.load(os.path.join("..", "gun5", "tfidf_vectorizer.pkl"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    print("Gelen veri:", data)

    if not data or 'text' not in data:
        return jsonify({"error": "No message provided"}), 400

    mesaj = data['text']
    mesaj_vec = vectorizer.transform([mesaj])
    tahmin = model.predict(mesaj_vec)[0]

    return jsonify({"prediction": tahmin})

if __name__ == '__main__':
    print("Flask API başlatılıyor...")
    app.run(debug=True)