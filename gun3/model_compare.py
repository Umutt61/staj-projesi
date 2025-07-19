import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Veri yükle
df = pd.read_csv("../gun2/sms_clean.csv")

# X (metin), y (etiket)
X = df['text']
y = df['label']

# Eğitim / doğrulama ayır (stratify = y → sınıflar dengeli dağılsın)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# Kullanılacak modeller
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "MultinomialNB": MultinomialNB(),
    "RandomForest": RandomForestClassifier(n_estimators=100)
}

# Sonuçlar
results = []

for name, model in models.items():
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_val_vec)

    acc = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds, pos_label="spam")
    rec = recall_score(y_val, preds, pos_label="spam")
    f1 = f1_score(y_val, preds, pos_label="spam")
    cm = confusion_matrix(y_val, preds)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "Confusion Matrix": cm.tolist()
    })

# Sonuçları yazdır
df_results = pd.DataFrame(results)
print(df_results)

# Kaydet
df_results.to_markdown("results_day3.md", index=False)