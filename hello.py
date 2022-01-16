from nltk.tokenize import word_tokenize
from greek_stemmer import GreekStemmer  # https://github.com/alup/python_greek_stemmer
import pandas as pd
import unicodedata
# from pandas.core.base import SpecificationError
# import numpy as np
from collections import Counter
import math

df = pd.read_csv("Greek_Parliament_Proceedings_1989_2020_DataSample.csv")
stemmer = GreekStemmer()


def remove_stopwords_and_stem(stop_words, speech_one):
    # print(speech_one)
    word_tokens = word_tokenize(speech_one)
    filtered_sentence = [w for w in word_tokens if (not w.lower() in stop_words) and w.isalnum()]

    speech_temp = []
    for w in filtered_sentence:
        # https://stackoverflow.com/a/62899722
        d = {ord('\N{COMBINING ACUTE ACCENT}'):None}
        word = unicodedata.normalize('NFD',w).upper().translate(d)
        speech_temp.append(stemmer.stem(word))
        # print(w, ':', stemmer.stem(word))
    return speech_temp


def count_words(s):
    counter = Counter(s)
    common_words = (counter.most_common())
    return common_words


def print_weights(common_words, s, target):
    for i in range(min(len(common_words), 3)):
        print(common_words[i][0], "TF-IDF:", tfidf(i, common_words, s, target))


def tf(i, common_words, s):
    return common_words[i][1] / len(s)


def idf(i, common_words, target):
    ni = 0
    if target is names:
        column = 'member_name'
    elif target is parties:
        column = 'political_party'
    else:
        target = []
        ni = sum(1 for sp in speech if common_words[i][0] in sp)

    for element in target:
        if any(common_words[i][0] in l for l in df[df[column] == element]['processed_speech']):
            ni += 1

    idf_ = math.log(len(speech)) / ni

    if idf_ < 0.0:
        return 0.0
    return idf_


def tfidf(i, common_words, s, target):
    return tf(i, common_words, s) * idf(i, common_words, target)


def main():
    stop_words_list = []
    with open('stopwords.txt', 'r', encoding='utf8') as filestream:
        for line in filestream:
            stop_words_list.append(line.lstrip().rstrip())
    stop_words = set(stop_words_list)

    global speech
    speech = [remove_stopwords_and_stem(stop_words, i) for i in df['speech']]
    df['processed_speech'] = [remove_stopwords_and_stem(stop_words, i) for i in df['speech']]

    global names
    names = df['member_name'].unique()

    global parties
    parties = df['political_party'].unique()

    # Per parliament member
    for name in names:
        print("=======", name, "======")
        temp_list1 = df[df['member_name'] == name]['processed_speech'].tolist()
        speech_of_member = [x for l1 in temp_list1 for x in l1]
        common_words = count_words(speech_of_member)
        print_weights(common_words, speech_of_member, names)

    # Per parties
    for party in parties:
        print("=======", party, "======")
        temp_list2 = df[df['political_party'] == party]['processed_speech']
        speech_of_party = [x for l2 in temp_list2 for x in l2]
        common_words = count_words(speech_of_party)
        print_weights(common_words, speech_of_party, parties)

    # Per speech
    for s in df['processed_speech']:
        common_words = count_words(s)
        print_weights(common_words, s, [])


if __name__ == "__main__":
    main()
