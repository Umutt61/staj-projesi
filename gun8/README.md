# SMS Spam Sınıflandırma (Gün 8)

## Problem Tanımı
Bu projede, SMS mesajlarının spam mi yoksa ham (normal) mı olduğunu sınıflandırmayı amaçlıyoruz. Ham mesajlar 0 ile spam mesajlar 1 etiketiyle gösterilmiştir.

## Hedef Değişken
- `label`: 0 = ham, 1 = spam

## Veri Seti
- Kaynak: `sms.tsv` (etiket, mesaj)
- Boyut: 5572 mesaj

## Kullandığım Araçlar
- Python, pandas, sklearn, seaborn
- TF-IDF + LogisticRegression

## İlk Sonuçlar (Baseline Model)
              precision    recall  f1-score   support

           0       0.97      1.00      0.98       966
           1       1.00      0.80      0.89       149

    accuracy                           0.97      1115
   macro avg       0.98      0.90      0.94      1115
weighted avg       0.97      0.97      0.97      1115

Accuracy: 0.9730941704035875
Precision: 1.0
Recall: 0.7986577181208053
F1-Score: 0.8880597014925373

> Not: Sonuçlar `baseline.py` dosyasındaki Logistic Regression modeliyle elde edilmiştir.