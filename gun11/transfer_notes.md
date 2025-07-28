# Gün 11 - Transfer Learning Notları

## Kullanılan Model
- Model: distilbert-base-uncased
- Kaynak: Hugging Face Transformers

## Giriş Verisi
- Metin: "Yapay zeka dünyayı değiştiriyor."

## Model Çıktısı
- `last_hidden_state` boyutu: [1, 17, 768]
- Yani model 17 token üretmiş ve her biri 768 boyutlu vektördür.

## Gözlem
- DistilBERT, küçük ve hızlı bir transformer modelidir.
- Çıktı, her kelimeye karşılık gelen vektörleri verir.
- Bu vektörler sınıflandırma, kategorize, duygu analizi vb. işler için temel olabilir.