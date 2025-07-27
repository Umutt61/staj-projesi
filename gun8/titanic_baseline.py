import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# 1. Veriyi yükle
df = pd.read_csv("C:/Users/umut/staj-projesi/gun8/data/train.csv")

# 2. Kategorik değişkenleri sayısal hale getir (örneğin "Sex")
df["Sex"] = LabelEncoder().fit_transform(df["Sex"])  # male:1, female:0

# 3. Eksik verileri doldur
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# 4. Kullanılacak özellikleri seç
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
X = df[features]
y = df["Survived"]

# 5. Sınıf dağılımını görselleştir
sns.countplot(x=y)
plt.title("Hayatta Kalma Dağılımı (0: Öldü, 1: Hayatta)")
plt.show()

# 6. Train-Test ayrımı
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 7. Model eğitimi
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 8. Metrikler
print("\n Classification Report:\n")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))

# 9. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Öldü", "Hayatta"], yticklabels=["Öldü", "Hayatta"])
plt.title("Confusion Matrix")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.show()