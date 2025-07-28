from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Model ve vectorizer yükle
MODEL_PATH = os.path.join("..", "gun5", "best_model.pkl")
VECTORIZER_PATH = os.path.join("..", "gun5", "tfidf_vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def preprocess_input(data):
    if not data or 'text' not in data:
        return None, jsonify({"error": "No message provided"}), 400
    return data['text'], None, None

def predict_class(text):
    text_vec = vectorizer.transform([text])
    return model.predict(text_vec)[0]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    print("Gelen veri:", data)

    text, error_response, status_code = preprocess_input(data)
    if error_response:
        return error_response, status_code

    prediction = predict_class(text)
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    print("Flask API başlatılıyor...")
    app.run(debug=True)