import pandas as pd
import zipfile
import urllib.request

# Veri bağlantısı
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
zip_path = "smsspam.zip"

# İndir
urllib.request.urlretrieve(url, zip_path)

# Zip'ten çıkar
with zipfile.ZipFile(zip_path, 'r') as zip_unz:
    zip_unz.extractall()

# Veriyi oku
df = pd.read_csv("SMSSpamCollection", sep="\t", header=None, names=["label", "text"])

# İlk 5 satırı göster
print(df.head())

# Kaç kayıt var?
print(f"Toplam veri: {len(df)}")

# Eksik değer var mı?
print(df.isnull().sum())

# Yinelenen kayıtları sil
df = df.drop_duplicates()
print(f"Tekil veri: {len(df)}")

# Etiketleri say
print(df['label'].value_counts())

# Temiz veriyi kaydet
df.to_csv("sms_clean.csv", index=False)