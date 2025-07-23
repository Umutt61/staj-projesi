import requests

def test_predict_ham():
    response = requests.post("http://127.0.0.1:5000/predict", json={"text": "Selam, nasılsın?"})
    assert response.status_code == 200
    assert response.json()["prediction"] in ["ham", "spam"]

def test_predict_spam():
    response = requests.post("http://127.0.0.1:5000/predict", json={"text": "Kredi kartı borcunuzu hemen ödeyin!"})
    assert response.status_code == 200
    assert response.json()["prediction"] in ["ham", "spam"]

def test_predict_empty():
    response = requests.post("http://127.0.0.1:5000/predict", json={})
    assert response.status_code == 400