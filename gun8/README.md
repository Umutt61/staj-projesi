Titanic: Yolcuların Hayatta Kalma Tahmini

Bu projede Titanic veri seti kullanılarak, yolcuların hayatta kalıp kalmadığını tahmin eden basit bir makine öğrenmesi modeli geliştirilmiştir.

> Amaç
Yolcu bilgilerini kullanarak (Age, Pclass, Sex, vs.) hayatta kalma durumunu (Survived) tahmin eden bir model eğitmek.

> Kullanılan Veriler
train.csv: Modelin eğitildiği veri.

test.csv: Tahmin yapılacak veri.

gender_submission.csv: Kaggle tarafından verilen örnek tahmin dosyası.

> Adımlar
Veriler yüklendi ve temizlendi (NaN değerler, Sex ve Embarked encoding).
Özellik seçimi yapıldı (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked).
TF-IDF gibi text vektörleştirme yerine Numerical + Categorical encoding uygulandı.
Baseline model olarak Logistic Regression kullanıldı.

> İlk Sonuçlar
Metric	Değer
Accuracy: 0.8044692737430168
Precision: 0.765625
Recall: 0.7101449275362319
F1-Score: 0.7368421052631579

> Dosya Yapısı
gun8/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
├── baseline.py
├── predict_test_set.py
└── README.md

> Notlar
Cabin, Name gibi sütunlar çıkarıldı çünkü çok eksik veya az anlamlıydı.
predict_test_set.py yalnızca test setine tahmin üretir.
Tahmin sonuç dosyası, predicted_submission.csv olarak kaydedilmiştir.