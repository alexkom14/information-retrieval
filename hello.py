from nltk.tokenize import word_tokenize
from greek_stemmer import GreekStemmer # https://github.com/alup/python_greek_stemmer
import pandas as pd
import unicodedata
from pandas.core.base import SpecificationError
import numpy as np
from collections import Counter
import math

df = pd.read_csv("Greek_Parliament_Proceedings_1989_2020_DataSample.csv")
speech = []
stemmer = GreekStemmer()

def remove_stopwords_and_stem(stop_words, speech_one):
    #print(speech_one)
    word_tokens = word_tokenize(speech_one)
    filtered_sentence = [w for w in word_tokens if (not w.lower() in stop_words) and w.isalnum()]

    speech_temp = []
    for w in filtered_sentence:
        # https://stackoverflow.com/a/62899722
        d = {ord('\N{COMBINING ACUTE ACCENT}'):None}
        word = unicodedata.normalize('NFD',w).upper().translate(d)
        speech_temp.append(stemmer.stem(word))
        #print(w, ':', stemmer.stem(word))
    return speech_temp

def tf(i):
    return common_words[i][1] / len(s)

def idf(i):
    res = math.log(len(speech)) / sum(1 for s in speech if common_words[i][0] in s)
    if res < 0.0:
        return 0.0
    return res

def tfidf(i):
    return tf(i) * idf(i)



stop_words_list = []
with open('stopwords.txt', 'r', encoding='utf8') as filestream:
    for line in filestream:
        stop_words_list.append(line.lstrip().rstrip())
stop_words = set(stop_words_list)

speech = [remove_stopwords_and_stem(stop_words, i ) for i in df['speech']]

for s in speech:
    counter = Counter(s)
    common_words = (counter.most_common())
    for i in range(min(len(common_words), 10)):
        print(common_words[i][0], "TF-IDF:", tfidf(i))


    
    



