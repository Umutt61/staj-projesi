import requests
import json

url = "http://127.0.0.1:5000/predict"

ornek_mesajlar = [
    "Bugün %50 indirim sizi bekliyor!",
    "Nasılsın? Akşam buluşalım mı?",
    "Kredi kartınız kullanıma kapatılacaktır, hemen arayın!",
    "Yarınki sınav iptal oldu.",
    "TEBRİKLER! 250.000 TL kazandınız! Ödülünüzü almak için hemen 0850 123 45 67 numarasını arayın!"
]

sonuclar = []

for mesaj in ornek_mesajlar:
    response = requests.post(url, json={"text": mesaj})
    tahmin = response.json()
    print(f"Mesaj: {mesaj} -> Tahmin: {tahmin}")
    sonuclar.append({"mesaj": mesaj, "tahmin": tahmin["prediction"]})

# JSON olarak kaydet
with open("pred_samples.json", "w", encoding="utf-8") as f:
    json.dump(sonuclar, f, ensure_ascii=False, indent=2)