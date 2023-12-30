from google.colab import drive
drive.mount ('/content/drive')

# mendefinisikan data path
%cd '/content/drive/MyDrive/D4 TEKNIK INFORMATIKA/SEMESTER 5/Pemrograman Sistem Cerdas 1/Pertemuan 2/Prak_Mohamad Idham Bahri/Implementasi Model/hasil_scrape'

!pip install google-play-scraper

from google_play_scraper import Sort, reviews_all, reviews
import re
import pandas as pd
import numpy as np
import datetime as dt
import string
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix, classification_report
# from google_play_scraper import Sort, reviews_all, reviews
import nltk
# import calendar
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from wordcloud import WordCloud

import pandas as pd
dframe=pd.read_csv("ulasan-rating5.csv")
print(dframe[1:4])

dframe.head()

from sklearn.utils.multiclass import unique_labels

unique_labels(dframe['score'])

#mengambil data dari satu kolom
data_ulasan=dframe['content']
print(data_ulasan[1:4])

# case folding
data_casefolding = data_ulasan.str.lower()
data_casefolding.head()

#filtering
import re
#url
filtering_url = [re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", ulasan) for ulasan in data_casefolding]
#cont
filtering_cont = [re.sub(r'\(cont\)'," ", ulasan)for ulasan in filtering_url]
#punctuatuion
filtering_punctuation = [re.sub('[!"”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]', ' ', ulasan) for ulasan in filtering_cont]  #hapus simbol'[!#?,.:";@()-_/\']'
#  hapus #tagger
filtering_tagger = [re.sub(r'#([^\s]+)', '', ulasan) for ulasan in filtering_punctuation]
#numeric
filtering_numeric = [re.sub(r'\d+', ' ', ulasan) for ulasan in filtering_tagger]
data_filtering = pd.Series(filtering_numeric)
print (data_filtering[:4])
#print (data_filtering) #jika ingin mencetak semua data_filtering

# tokenization menggunakan word_tokenize
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

data_tokens = [word_tokenize(line) for line in data_filtering]
print(data_tokens)

!pip install sastrawi

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# slang word
path_dataslang = open("kamus kata bakuu 1.csv")
dataslang = pd.read_csv(path_dataslang, encoding = 'utf-8', header=None, sep=";")

def replaceSlang(word):
  if word in list(dataslang[0]):
    indexslang = list(dataslang[0]).index(word)
    return dataslang[1][indexslang]
  else:
    return word

ulasan_formal = []
for data in data_katapenting:
  data_clean = [replaceSlang(word) for word in data]
  ulasan_formal.append(data_clean)
len_ulasan_formal = len(ulasan_formal)
print(ulasan_formal)
len_ulasan_formal

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
ind_stemmer = factory.create_stemmer()
def stemmer(line):
  temp = list()
  for word in line:
    word = ind_stemmer.stem(word)
    temp.append(word)
  return temp

ulasan_dasar = [stemmer (line) for line in ulasan_formal]
print(ulasan_dasar)

# Feature Extraction: Bag of Words (tf-idf)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
corpus = [' '.join(words) for words in ulasan_dasar]
# X = vectorizer.fit_transform(corpus)

# saran bu intan
vektor_tfidf = vectorizer.fit_transform(corpus)
vektor_tfidf = vektor_tfidf.toarray()
vektor_tfidf.shape

import pickle

with open('reviews.pkl', 'wb') as model_file:
    pickle.dump(corpus, model_file)

corpus

# Modeling: Support Vector Machine (SVM)
from sklearn.model_selection import train_test_split

y = dframe['score']
X_train, X_test, y_train, y_test = train_test_split(vektor_tfidf, y, test_size=0.2, random_state=42)

from sklearn.svm import SVC

svm_model = SVC()
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Evaluasi Model
y_pred = svm_model.predict(X_test)

# Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)

# Visualisasi Confusion Matrix dengan Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=svm_model.classes_, yticklabels=svm_model.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Evaluasi Akurasi
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

import pickle
import joblib
joblib.dump(svm_model, 'revisi_hasil_sentimen.pkl')