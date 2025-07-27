import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Test verisini yüklüyoruz
test_df = pd.read_csv("C:/Users/umut/staj-projesi/gun8/data/test.csv")

# Aynı şekilde preprocessing
test_df["Sex"] = LabelEncoder().fit_transform(test_df["Sex"])
test_df["Age"].fillna(test_df["Age"].median(), inplace=True)
test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
X_test = test_df[features]

# Eğitim verisiyle aynı şekilde eğitilmiş modeli yeniden eğitiyoruz
train_df = pd.read_csv("C:/Users/umut/staj-projesi/gun8/data/train.csv")
train_df["Sex"] = LabelEncoder().fit_transform(train_df["Sex"])
train_df["Age"].fillna(train_df["Age"].median(), inplace=True)
train_df["Embarked"].fillna(train_df["Embarked"].mode()[0], inplace=True)
X_train = train_df[features]
y_train = train_df["Survived"]

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Tahmin yap
y_pred = model.predict(X_test)

# Sonuçları gender_submission.csv formatında kaydet
submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": y_pred
})
submission.to_csv("C:/Users/umut/staj-projesi/gun8/data/predicted_submission.csv", index=False)

print(" Tahminler başarıyla kaydedildi: predicted_submission.csv")