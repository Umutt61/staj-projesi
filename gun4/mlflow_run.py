import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Veri yükleme
df = pd.read_csv("../gun2/sms_clean.csv")
X = df['text']
y = df['label']

# Eğitim / doğrulama ayır
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# Modeller
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "MultinomialNB": MultinomialNB(),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# MLflow ile deneme
mlflow.set_experiment("sms_model_compare")

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_val_vec)

        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, pos_label="spam")
        rec = recall_score(y_val, preds, pos_label="spam")
        f1 = f1_score(y_val, preds, pos_label="spam")

        # MLflow ile metrikleri kaydetme
        mlflow.log_param("model_name", name)
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        print(f"{name} -> Accuracy: {acc:.4f}, F1: {f1:.4f}")