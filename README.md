# Web Scraping Language Detector 🌍

This project is a machine learning application that collects real-time text data using web scraping and develops a multilingual language detection system.

## 🎯 Project Overview

This system automatically collects text data from news websites in 9 different languages, cleans it, and uses this data to train a language detection model. The trained model can accurately detect which language new texts are written in.

## 🌐 Supported Languages

- 🇹🇷 **Turkish** (Türkçe)
- 🇺🇸 **English** (İngilizce)
- 🇩🇪 **German** (Almanca)
- 🇫🇷 **French** (Fransızca)
- 🇪🇸 **Spanish** (İspanyolca)
- 🇷🇺 **Russian** (Rusça)
- 🇨🇳 **Chinese** (Çince)
- 🇰🇷 **Korean** (Korece)
- 🇯🇵 **Japanese** (Japonca)

## 🚀 Features

### Web Scraping

- Real-time data collection from news websites
- Smart HTML parsing and cleaning
- Ethical web scraping with rate limiting
- Error handling and reliable data collection

### Data Processing

- Automatic text cleaning and preprocessing
- Feature extraction with TF-IDF vectorization
- Character n-gram analysis (1-3 grams)
- Data quality control

### Machine Learning

- Multinomial Naive Bayes classifier
- Stratified train-test split
- Detailed performance evaluation
- Model saving and loading

### Interactive Testing

- Real-time language detection testing
- Prediction reliability with confidence scores
- User-friendly interface

## 📋 Requirements

### Python Libraries

```bash
pip install pandas numpy matplotlib seaborn
pip install scikit-learn nltk
pip install requests beautifulsoup4
pip install pickle
```

### System Requirements

- Python 3.7+
- Internet connection (for web scraping)
- At least 4GB RAM (for large datasets)

## 🛠️ Installation

1. **Clone the project:**

```bash
git clone <repository-url>
cd new3
```

2. **Install required libraries:**

```bash
pip install -r requirements.txt
```

3. **Run the project:**

```bash
python dl.py
```

## 📊 Usage

### Basic Usage

```python
from dl import WebScrapingLanguageDetector

# Create detector
detector = WebScrapingLanguageDetector()

# Collect data and train model
data = detector.collect_data_from_web()
detector.prepare_data()
x_train, x_test, y_train, y_test = detector.extract_features()
detector.train_model(x_train, y_train)

# Evaluate model
accuracy = detector.evaluate_model(x_test, y_test)

# Save model
detector.save_model()
```

### Language Detection

```python
# Language detection for single text
text = "Hello, how are you today?"
language, confidence = detector.predict_language(text)
print(f"Language: {language}, Confidence: {confidence:.2f}")
```

### Interactive Testing

```python
# Interactive test mode
detector.interactive_test()
```

## 🔧 Configuration

### Web Scraping Settings

```python
# Maximum number of texts per language
max_texts_per_language = 30

# Maximum number of texts per URL
max_texts_per_url = 10

# Wait time between requests (seconds)
time.sleep(random.uniform(2, 4))
```

### Model Parameters

```python
# TF-IDF Vectorizer settings
vectorizer = TfidfVectorizer(
    analyzer='char',           # Character-based analysis
    ngram_range=(1,3),         # 1-3 character n-grams
    max_features=2000,         # Maximum number of features
    min_df=2,                  # Minimum document frequency
    max_df=0.95                # Maximum document frequency
)

# Naive Bayes settings
model = MultinomialNB(alpha=0.1)  # Smoothing parameter
```

## 📈 Performance

### Model Performance

- **Accuracy:** 85-95% (depending on data quality)
- **Training Time:** 1-3 minutes
- **Prediction Time:** <1 second
- **Supported Text Length:** 20-500 characters

### Web Scraping Performance

- **Total Data Collection Time:** 5-10 minutes
- **Average Number of Texts:** 200-300 texts
- **Success Rate:** 90%+ (depending on site accessibility)

## 🗂️ File Structure

```
new3/
├── dl.py                              # Main application file
├── web_scraped_language_detector.pkl  # Trained model
├── requirements.txt                   # Python requirements
└── README.md                          # This file
```

## 🔍 Technical Details

### Data Collection Process

1. **URL List:** 3 news websites per language
2. **HTML Parsing:** Content extraction with BeautifulSoup
3. **Text Filtering:** Texts between 20-500 characters
4. **Cleaning:** Removal of script, style, nav elements

### Feature Extraction

- **TF-IDF Vectorization:** Converting texts to numerical vectors
- **Character N-grams:** 1-3 character combinations
- **Feature Selection:** Selection of the most important 2000 features

### Model Training

- **Algorithm:** Multinomial Naive Bayes
- **Cross-validation:** Balanced distribution with stratified split
- **Hyperparameter:** Alpha=0.1 smoothing

## 🚨 Important Notes

### Ethical Usage

- Respect websites' robots.txt files
- Don't overload servers by using rate limiting
- Don't use copyrighted content for commercial purposes

### Error Handling

- Automatic retry on internet connection interruptions
- Alternative URLs for inaccessible sites
- Data quality control and filtering

### Performance Optimization

- Batch processing for large datasets
- Fast reloading with model caching
- Memory-efficient data structures

## 🔮 Future Enhancements

- [ ] Support for more languages
- [ ] Deep learning models (LSTM, BERT)
- [ ] Web interface development
- [ ] API endpoints
- [ ] Real-time streaming analysis
- [ ] Multi-model ensemble

## 🤝 Contributing

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 📞 Contact

For questions about the project:

- **Email:** [your-email@example.com]
- **GitHub:** [your-github-username]

## 🙏 Acknowledgments

- **scikit-learn** - Machine learning library
- **BeautifulSoup** - HTML parsing
- **requests** - HTTP requests
- **pandas** - Data manipulation

---

⭐ **If you liked this project, don't forget to give it a star!**
