# Fine-Tuning Nedir?  

Fine-tuning, önceden eğitilmiş (pre-trained) büyük bir yapay zeka modelinin (ör. BERT, GPT, LLaMA, Stable Diffusion) farklı bir veri seti üzerinde tekrar eğitilerek belirli bir göreve uyarlanması işlemidir.  

Bu sayede sıfırdan model eğitmek yerine mevcut bilgi temeli korunur ve daha az veriyle, daha kısa sürede, belirli bir probleme odaklı yüksek performanslı modeller elde edilir.  

---

## Neden Fine-Tuning Kullanılır?  

- **Zaman ve maliyet tasarrufu**: Devasa veri setlerine ihtiyaç duyulmaz.  
- **Özel alan bilgisi**: Tıp, hukuk, finans gibi dar alanlarda daha başarılı sonuçlar.  
- **Daha yüksek doğruluk**: Göreve odaklanmış optimize edilmiş performans.  
- **Esneklik**: Model, farklı diller, domainler veya kullanım senaryolarına kolayca uyarlanabilir.  

---

## Fine-Tuning Teknikleri  

Fine-tuning yöntemleri temelde ikiye ayrılır:  

### 1. **Klasik Yöntemler**  

- **Full Fine-Tuning**: Tüm model parametreleri yeniden eğitilir.  
- **Feature Extraction**: Önceden öğrenilmiş katmanlar sabitlenir, sadece son katman eğitilir.  
- **Layer Freezing**: Bazı katmanlar dondurulur, diğerleri eğitilir.  

### 2. **Parameter-Efficient Fine-Tuning (PEFT)**  

Son yıllarda büyük dil modelleri (LLM) için geliştirilmiş modern tekniklerdir. Burada amaç, milyarlarca parametreyi yeniden eğitmek yerine sadece küçük ek parametrelerle verimli bir şekilde ince ayar yapmaktır.  

- **LoRA (Low-Rank Adaptation)**  
  - Model ağırlıklarına düşük boyutlu matrisler eklenir.  
  - Kaynak tüketimi çok düşüktür.  
  - Hugging Face PEFT kütüphanesi ile yaygın olarak kullanılır.  

- **QLoRA (Quantized LoRA)**  
  - LoRA tekniğinin daha verimli versiyonudur.  
  - Modeli düşük hassasiyet (4-bit quantization) ile yükleyip LoRA katmanları ekler.  
  - Çok daha küçük GPU’larda bile büyük LLM’lerin fine-tune edilmesini mümkün kılar.  

- **Adapter Layers**  
  - Orijinal modele ek küçük katmanlar yerleştirilir.  
  - Sadece bu adapter katmanları eğitilir.  

- **Prefix Tuning / Prompt Tuning**  
  - Modele eklenen sanal “prompt vektörleri” eğitilir.  
  - Modelin ana ağırlıkları değiştirilmez.  

---

## Modern Araçlar ve Platformlar  

Fine-tuning sadece teorik tekniklerden ibaret değildir; günümüzde bu iş için pratik araçlar geliştirilmiştir:  

- **Hugging Face Transformers + PEFT** → LoRA/QLoRA ile fine-tuning yapılabilir.  
- **Ollama**  
  - LLaMA, Mistral, Gemma gibi açık kaynak LLM’leri kolayca çalıştırmaya yarar.  
  - `Modelfile` kullanarak prompt-template, hyperparameter ayarları ve dataset ile **custom LLM fine-tuning** yapılabilir.  
  - Geliştiricilere model deploy etme ve paylaşma kolaylığı sağlar.  
- **OpenAI Fine-Tuning API** → GPT-3.5 ve GPT-4 için özel veriyle fine-tune imkanı.  
- **LangChain / LlamaIndex** → Fine-tuning yerine *RAG (Retrieval-Augmented Generation)* yaklaşımı kullanarak dış veriyi modele bağlama.  
- **DeepSpeed & ColossalAI** → Büyük modeller için optimizasyonlu fine-tuning çözümleri.  

---

## Fine-Tuning Süreci  

1. **Model Seçimi** → Uygun pre-trained model belirlenir (örn. LLaMA2, BERT, ResNet).  
2. **Veri Toplama** → Alanla ilgili doğru ve temiz veri hazırlanır.  
3. **Ön İşleme** → Tokenizasyon, normalizasyon, etiketleme.  
4. **Fine-Tuning Yöntemi Seçimi** → Full, LoRA, QLoRA, Adapter vs.  
5. **Eğitim** → Model belirlenen teknikle eğitilir.  
6. **Değerlendirme** → Accuracy, F1-score, loss gibi metrikler ölçülür.  
7. **Deploy** → Eğitim sonrası model gerçek kullanım senaryosuna entegre edilir (örn. Ollama, Hugging Face Hub).  

---

## Uygulama Alanları  

- **NLP** → Chatbot, metin sınıflandırma, özetleme  
- **Bilgisayarla Görü** → Nesne tanıma, tıbbi görüntü analizi  
- **Ses İşleme** → Konuşma tanıma, sesli asistan  
- **Alan Spesifik Modeller** → Medikal rapor analizi, hukuk belgeleri sınıflandırma  

---

## Sonuç  

Fine-tuning, yapay zeka modellerini gerçek dünyadaki belirli problemlere uyarlamanın en güçlü yollarından biridir.  
Klasik yöntemlerin yanında, **LoRA, QLoRA, PEFT ve Ollama gibi modern yaklaşımlar** sayesinde artık çok daha verimli ve düşük maliyetli bir şekilde büyük modelleri kişiselleştirmek mümkündür.  

---