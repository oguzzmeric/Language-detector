# Web Scraping Language Detector 🌍

Bu proje, web scraping kullanarak gerçek zamanlı metin verilerini toplayan ve çok dilli dil tanıma sistemi geliştiren bir makine öğrenmesi uygulamasıdır.

## 🎯 Proje Özeti

Bu sistem, 9 farklı dildeki haber sitelerinden metin verilerini otomatik olarak toplar, temizler ve bu verileri kullanarak bir dil tanıma modeli eğitir. Eğitilen model, yeni metinlerin hangi dilde yazıldığını yüksek doğrulukla tespit edebilir.

## 🌐 Desteklenen Diller

- 🇹🇷 **Türkçe** (Turkish)
- 🇺🇸 **İngilizce** (English)
- 🇩🇪 **Almanca** (German)
- 🇫🇷 **Fransızca** (French)
- 🇪🇸 **İspanyolca** (Spanish)
- 🇷🇺 **Rusça** (Russian)
- 🇨🇳 **Çince** (Chinese)
- 🇰🇷 **Korece** (Korean)
- 🇯🇵 **Japonca** (Japanese)

## 🚀 Özellikler

### Web Scraping

- Gerçek zamanlı haber sitelerinden veri toplama
- Akıllı HTML parsing ve temizleme
- Rate limiting ile etik web scraping
- Hata yönetimi ve güvenilir veri toplama

### Veri İşleme

- Otomatik metin temizleme ve ön işleme
- TF-IDF vektörizasyonu ile özellik çıkarımı
- Karakter n-gram analizi (1-3 gram)
- Veri kalitesi kontrolü

### Makine Öğrenmesi

- Multinomial Naive Bayes sınıflandırıcı
- Stratified train-test split
- Detaylı performans değerlendirmesi
- Model kaydetme ve yükleme

### Etkileşimli Test

- Gerçek zamanlı dil tanıma testi
- Güven skoru ile tahmin güvenilirliği
- Kullanıcı dostu arayüz

## 📋 Gereksinimler

### Python Kütüphaneleri

```bash
pip install pandas numpy matplotlib seaborn
pip install scikit-learn nltk
pip install requests beautifulsoup4
pip install pickle
```

### Sistem Gereksinimleri

- Python 3.7+
- İnternet bağlantısı (web scraping için)
- En az 4GB RAM (büyük veri setleri için)

## 🛠️ Kurulum

1. **Projeyi klonlayın:**

```bash
git clone <repository-url>
cd new3
```

2. **Gerekli kütüphaneleri yükleyin:**

```bash
pip install -r requirements.txt
```

3. **Projeyi çalıştırın:**

```bash
python dl.py
```

## 📊 Kullanım

### Temel Kullanım

```python
from dl import WebScrapingLanguageDetector

# Detector oluştur
detector = WebScrapingLanguageDetector()

# Veri topla ve modeli eğit
data = detector.collect_data_from_web()
detector.prepare_data()
x_train, x_test, y_train, y_test = detector.extract_features()
detector.train_model(x_train, y_train)

# Modeli değerlendir
accuracy = detector.evaluate_model(x_test, y_test)

# Modeli kaydet
detector.save_model()
```

### Dil Tanıma

```python
# Tek metin için dil tanıma
text = "Hello, how are you today?"
language, confidence = detector.predict_language(text)
print(f"Dil: {language}, Güven: {confidence:.2f}")
```

### Etkileşimli Test

```python
# Etkileşimli test modu
detector.interactive_test()
```

## 🔧 Yapılandırma

### Web Scraping Ayarları

```python
# Her dil için maksimum metin sayısı
max_texts_per_language = 30

# Her URL için maksimum metin sayısı
max_texts_per_url = 10

# İstekler arası bekleme süresi (saniye)
time.sleep(random.uniform(2, 4))
```

### Model Parametreleri

```python
# TF-IDF Vectorizer ayarları
vectorizer = TfidfVectorizer(
    analyzer='char',           # Karakter bazlı analiz
    ngram_range=(1,3),         # 1-3 karakter n-gram
    max_features=2000,         # Maksimum özellik sayısı
    min_df=2,                  # Minimum doküman frekansı
    max_df=0.95                # Maksimum doküman frekansı
)

# Naive Bayes ayarları
model = MultinomialNB(alpha=0.1)  # Smoothing parametresi
```

## 📈 Performans

### Model Performansı

- **Doğruluk:** %85-95 (veri kalitesine bağlı)
- **Eğitim Süresi:** 1-3 dakika
- **Tahmin Süresi:** <1 saniye
- **Desteklenen Metin Uzunluğu:** 20-500 karakter

### Web Scraping Performansı

- **Toplam Veri Toplama Süresi:** 5-10 dakika
- **Ortalama Metin Sayısı:** 200-300 metin
- **Başarı Oranı:** %90+ (site erişilebilirliğine bağlı)

## 🗂️ Dosya Yapısı

```
new3/
├── dl.py                              # Ana uygulama dosyası
├── web_scraped_language_detector.pkl  # Eğitilmiş model
├── requirements.txt                   # Python gereksinimleri
└── README.md                          # Bu dosya
```

## 🔍 Teknik Detaylar

### Veri Toplama Süreci

1. **URL Listesi:** Her dil için 3 haber sitesi
2. **HTML Parsing:** BeautifulSoup ile içerik çıkarımı
3. **Metin Filtreleme:** 20-500 karakter arası metinler
4. **Temizleme:** Script, style, nav elementlerinin kaldırılması

### Özellik Çıkarımı

- **TF-IDF Vektörizasyonu:** Metinleri sayısal vektörlere dönüştürme
- **Karakter N-gram:** 1-3 karakter kombinasyonları
- **Özellik Seçimi:** En önemli 2000 özelliğin seçilmesi

### Model Eğitimi

- **Algoritma:** Multinomial Naive Bayes
- **Cross-validation:** Stratified split ile dengeli dağılım
- **Hyperparameter:** Alpha=0.1 smoothing

## 🚨 Önemli Notlar

### Etik Kullanım

- Web sitelerinin robots.txt dosyalarına saygı gösterin
- Rate limiting kullanarak sunucuları aşırı yüklemeyin
- Telif hakkı olan içerikleri ticari amaçla kullanmayın

### Hata Yönetimi

- İnternet bağlantısı kesintilerinde otomatik yeniden deneme
- Erişilemeyen siteler için alternatif URL'ler
- Veri kalitesi kontrolü ve filtreleme

### Performans Optimizasyonu

- Büyük veri setleri için batch processing
- Model caching ile hızlı yeniden yükleme
- Memory-efficient veri yapıları

## 🔮 Gelecek Geliştirmeler

- [ ] Daha fazla dil desteği
- [ ] Deep learning modelleri (LSTM, BERT)
- [ ] Web arayüzü geliştirme
- [ ] API endpoint'leri
- [ ] Gerçek zamanlı streaming analizi
- [ ] Çoklu model ensemble

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Commit yapın (`git commit -m 'Add some AmazingFeature'`)
4. Push yapın (`git push origin feature/AmazingFeature`)
5. Pull Request açın

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 📞 İletişim

Proje hakkında sorularınız için:

- **Email:** [your-email@example.com]
- **GitHub:** [your-github-username]

## 🙏 Teşekkürler

- **scikit-learn** - Makine öğrenmesi kütüphanesi
- **BeautifulSoup** - HTML parsing
- **requests** - HTTP istekleri
- **pandas** - Veri manipülasyonu

---

⭐ **Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!**
