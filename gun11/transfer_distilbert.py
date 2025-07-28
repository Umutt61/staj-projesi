from transformers import AutoTokenizer, AutoModel
import torch

# Model ve tokenizer'ı yükle
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Örnek bir metin
text = "Yapay zeka dünyayı değiştiriyor."

# Tokenize et
inputs = tokenizer(text, return_tensors="pt")

# Modeli çalıştır
with torch.no_grad():
    outputs = model(**inputs)

# Embedding'leri al
last_hidden_state = outputs.last_hidden_state
print("Embedding çıktısı şekli:", last_hidden_state.shape)