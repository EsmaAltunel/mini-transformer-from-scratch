 # **🎸 Swiftie-GPT: From-Scratch Transformer**

Bu proje, Taylor Swift'in lirik dünyasını modellemek için PyTorch kullanılarak sıfırdan inşa edilmiş bir Decoder-only Transformer mimarisidir. Hazır model (GPT, Llama vb.) kullanmak yerine; tokenizer'dan dikkat mekanizmasına kadar her bileşen manuel olarak kodlanmıştır.


## **🧠 Teknik Mimari**

### **1. Hybrid Tokenizer & Dataset**

* **Greedy Tokenization:** Regex tabanlı ([a-zA-Z']+|[0-9]+|[?!,.]|\s+) hibrit bir yapı kullanıldı. Bilinen kelimeleri tam, bilinmeyenleri karakter bazlı işleyerek esneklik sağlar.

* **Sliding Window:** Veri seti, her adımda bir token kaydırarak modeli bir sonraki karakteri tahmin etmeye zorlayan (x, y) çiftleri üretir.

### **2. Sinusoidal Embedding**

* **Positional Encoding:** Kelime sırasını anlamak için eğitilebilir embedding yerine sabit sinüs/kosinüs dalgaları kullanılmıştır. Bu, modelin uzun dizilerdeki zamansal ilişkiyi matematiksel bir hassasiyetle kavramasını sağlar.

### **3. Causal Multi-Head Attention**

* **Self-Attention:** Kelimeler arası anlamsal bağları Q,K,V matrisleri üzerinden çözer.

* **Masking:** torch.tril ile modelin eğitim sırasında "geleceği görmesi" engellenmiştir.

* **Multi-Head:** 4 paralel kafa ile metnin farklı anlamsal boyutlarına (kafiye, özne-yüklem vb.) aynı anda odaklanır.

### **4. Gated Decoder & MLP**

* **Gated Projection:** Klasik MLP yerine Llama 3 tarzı "Gated Linear Unit" ve GeLU aktivasyonu kullanılmıştır. Bu "kapı" mekanizması modelin öğrenme kapasitesini artırır.

* **Manual LayerNorm:** Stabil bir eğitim için normalizasyon katmanı sıfırdan matematiksel formülüyle kodlanmıştır.

### **5. Generation**

* **Top-K & Temperature:** Üretim sırasında "yaratıcılık" ayarı yapılır. Top-K ile saçma ihtimaller elenirken, Temperature ile modelin risk alma seviyesi (yaratıcılığı) belirlenir.

## **⚙️ Eğitim ve Hiper-Parametreler**

### **Parametre Değer Açıklama**

* **Context Length** 128 Modelin bir seferde baktığı karakter penceresi

* **Batch Size** 16 Her adımda işlenen örnek sayısı

* **Embedding Dim** 128 Kelimelerin temsil edildiği vektör boyutu

* **Num Heads** 4 Multi-head attention kafa sayısı

* **Num Layers** 6 Üst üste binen Decoder bloğu sayısı

* **Learning Rate** 5e-4 AdamW optimizer öğrenme oranı

* **Epochs** 100 Toplam eğitim tur sayısı

  * **Checkpointing:** Eğitim sonunda model, sözlük ve konfigürasyonla birlikte model.pth olarak kaydedilir.

  * **Visuals:** Eğitim süreci izlenerek hata payını gösteren loss_curve.png grafiği üretilir.

## **🚀 Kullanıcı Arayüzü**

Modeli test etmek için Gradio tabanlı modern bir web arayüzü sunulmuştur:

* **Temperature:** Modelin risk alma/yaratıcılık seviyesini ayarlar.

* **Top-K:** En yüksek olasılıklı k kelime arasından seçim yaparak tutarlılığı korur. 
