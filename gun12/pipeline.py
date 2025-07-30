from transformers import pipeline
import time

# Pipeline'ı başlatma
classifier = pipeline("text-classification")

# Test metinleri
texts = [
    "I love this movie! It was amazing and full of emotions.",
    "This product is terrible. I'm never buying it again.",
    "The service was okay, but could have been better."
]

# Sonuçları toplama
results = []

for text in texts:
    start_time = time.time()
    output = classifier(text)
    end_time = time.time()
    duration = round(end_time - start_time, 3)

    results.append({
        "input": text,
        "output": output[0]['label'],
        "score": round(output[0]['score'], 2),
        "time": duration
    })

# Yazdırma
for r in results:
    print(f"Input: {r['input']}\nOutput: {r['output']} (Score: {r['score']}) | Time: {r['time']}s\n")