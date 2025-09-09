import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import nltk
import time
import random
import re
import warnings
import pickle
import os
import requests
from bs4 import BeautifulSoup
warnings.filterwarnings('ignore')

class WebScrapingLanguageDetector:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.vectorized = None
        self.model  = None
        self.data = None
    
    def get_multilingual_sources(self):
        return{
            'Turkish': [
                'https://www.haberturk.com',
                'https://www.milliyet.com.tr',
                'https://www.hurriyet.com.tr'
            ],
            'English': [
                'https://www.bbc.com/news',
                'https://www.cnn.com',
                'https://www.reuters.com'
            ],
            'German': [
                'https://www.spiegel.de',
                'https://www.zeit.de',
                'https://www.faz.net'
            ],
            'French': [
                'https://www.lemonde.fr',
                'https://www.liberation.fr',
                'https://www.lefigaro.fr'
            ],
            'Spanish': [
                'https://www.elpais.com',
                'https://www.elmundo.es',
                'https://www.abc.es'
            ],
            'Russian': [
                'https://www.rbc.ru',
                'https://www.rt.com',
                'https://www.tass.ru'
            ],
            'Chinese': [
                'https://www.xinhuanet.com',
                'https://www.people.com.cn',
                'https://www.zaobao.com'
            ],
            'Korean': [
                'https://www.yna.co.kr',
                'https://www.hankookilbo.com',
                'https://www.khan.co.kr'
            ],
            'Japanese': [
                'https://www.nikkei.com',
                'https://www.asahi.com',
                'https://www.jiji.com'
            ]
        }
    
    def scrape_website(self,url,max_texts=10):
        try:
            print(f"{url}işleniyor")
            response = self.session.get(url,timeout=10)
            soup = BeautifulSoup(response.content,'html.parser')

            for tag in soup(['script','style','nav','footer','header','aside']):
                tag.decompose()
            
            texts = []

            for tag in soup.find_all(['p','h1','h2','h3','div']):
                text = tag.get_text().strip()
                if len(text) > 20 and len(text) < 500:
                    texts.append(text)
            
            return texts[:max_texts]

        except Exception as e:
            print(f"{url} hatası: {e}")
            return []
        
    
    def collect_data_from_web(self,max_texts_per_language=30):
        print("webden weri alıyoruz")

        sources = self.get_multilingual_sources()
        collected_data = {'text': [], 'language': []}

        for language,urls in sources.items():
            print(f"\n{language} verisi toplanıyor...")

            language_texts = []

            for url in urls:
                if len(language_texts) >= max_texts_per_language:
                    break

                texts = self.scrape_website(url)

                for text in texts:
                    if len(language_texts) < max_texts_per_language:
                        language_texts.append(text)
                        collected_data['text'].append(text)
                        collected_data['language'].append(language)
                
                time.sleep(random.uniform(2,4))
            
            print(f"✅ {language}: {len(language_texts)} metin toplandı")

        self.data = pd.DataFrame(collected_data)
        print(f"\n🎉 Toplam {len(self.data)} metin toplandı!")
        return self.data

    def preprocess_text(self,text):
        if pd.isna(text):
            return " "
        
        text = str(text).lower()
        text = re.sub(r'[^\w\s]','',text)
        text = re.sub(r'\s+',' ',text)
        text = text.strip()
        return text
    
    def prepare_data(self):
        print("veri temizleniyor")

        self.data['cleaned_text'] = self.data['text'].apply(self.preprocess_text)
        self.data = self.data[self.data['cleaned_text'].str.len()>0]
        print(f"✅ Temizlenmiş veri: {len(self.data)} metin")

        return self.data

    def extract_features(self):
        print("özellik çıkarımı yapılıyor")
        x_train,x_test,y_train,y_test = train_test_split(
            self.data['cleaned_text'],
            self.data['language'],
            test_size = 0.2,
            random_state = 42,
            stratify = self.data['language']
        )

        self.vectorizer = TfidfVectorizer(
            analyzer = 'char',
            ngram_range = (1,3),
            max_features = 2000,
            min_df = 2,
            max_df = 0.95
        )
        x_train_vec = self.vectorizer.fit_transform(x_train)
        x_test_vec = self.vectorizer.transform(x_test)

        return x_train_vec,x_test_vec,y_train,y_test
    
    def train_model(self,x_train,y_train):

        self.model = MultinomialNB(alpha=0.1)
        self.model.fit(x_train,y_train)

        return self.model
    
    def evaluate_model(self,x_test,y_test):
        print("model değerlendiriliyor")

        y_pred = self.model.predict(x_test)
        accuracy = accuracy_score(y_test,y_pred)
        print(f"✅ Doğruluk: {accuracy:.3f}")
        print("\n📋 Detaylı rapor:")
        print(classification_report(y_test,y_pred))
        return accuracy
    
    def predict_language(self,text):
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model henüz eğitilmemiş!")
        
        cleaned_text = self.preprocess_text(text)
        text_vec = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(text_vec)[0]
        probability = self.model.predict_proba(text_vec).max()
        return prediction,probability
    
    def save_model(self,filename="web_scraped_language_detector.pkl"):
        model_data = {
            'model':self.model,
            'vectorizer':self.vectorizer,
            'languages':self.data['language'].unique().tolist()
        }
        with open(filename,'wb') as f:
            pickle.dump(model_data,f)
        print(f"✅ Model kaydedildi: {filename}")
        return filename
    
    def load_model(self,filename="web_scraped_language_detector.pkl"):
        with open(filename,'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        print(f"✅ Model yüklendi: {filename}")
    
    def interactive_test(self):
        print("\n🎮 Etkileşimli dil tanıma testi!")
        print("Çıkmak için 'quit' yazın\n")
        
        while True:
            text = input("Metin girin: ")
            if text.lower() == 'quit':
                break
            
            try:
                language,confidence = self.predict_language(text)
                print(f"🌍 Tespit edilen dil: {language}")
                print(f"🎯 Güven: {confidence:.2f}")
                print("-" * 40)
            except Exception as e:
                print(f"❌ Hata: {e}")
    
def main():
        detector = WebScrapingLanguageDetector()
        data = detector.collect_data_from_web()

        detector.prepare_data()
        x_train,x_test,y_train,y_test = detector.extract_features()
        detector.train_model(x_train,y_train)

        accuracy = detector.evaluate_model(x_test,y_test)
        detector.save_model()
        detector.interactive_test()
        print("sistem hazır")        
        
if __name__ == "__main__":
    main()
