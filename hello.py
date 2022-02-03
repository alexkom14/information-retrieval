from nltk.tokenize import word_tokenize
from greek_stemmer import GreekStemmer  # https://github.com/alup/python_greek_stemmer
import pandas as pd
import unicodedata
# from pandas.core.base import SpecificationError
import numpy as np
from numpy import dot
from numpy.linalg import norm
from collections import Counter
import math
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

np.set_printoptions(threshold=np.inf)
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


def tf(word_count, s):
    return word_count / len(s)


# PARTY AND MEMBER
def idf(i, common_words, target):
    ni = 0
    if target is names:
        column = 'member_name'
    elif target is parties:
        column = 'political_party'
    else:
        target = []

    for element in target:
        if any(common_words[i][0] in l for l in df[df[column] == element]['processed_speech']):
            ni += 1

    idf_ = math.log(len(speech)) / ni

    if idf_ < 0.0:
        return 0.0
    return idf_


def idf_simple(word):
    ni = sum(1 for sp in speech if word in sp)
    idf_ = math.log(len(speech)) / ni

    if idf_ < 0.0:
        return 0.0
    return idf_


def tfidf(i, common_words, s, target):
    return tf(i, common_words, s) * idf(i, common_words, target)

'''
def build_index():
    inverted_index = {}
    for doc, s in enumerate(df['processed_speech']):
        common_words = count_words(s)
        for w in common_words:
            word = w[0]
            word_count = w[1]
            if word not in inverted_index.keys():
                idf_ = idf_simple(word)
                inverted_index[word] = [[idf_], [doc, tf(word_count, s) * idf_]]
            else:
                inverted_index[word].append([doc, tf(word_count, s) * idf_])
    return inverted_index
'''
def build_index():
    inverted_index = {}
    for doc, s in enumerate(df['processed_speech']):
        common_words = count_words(s)
        for w in common_words:
            word = w[0]
            if word not in inverted_index.keys():
                inverted_index[word] = [doc]
            else:
                inverted_index[word].append(doc)
    return inverted_index

def make_query(stop_words, tfidf_vectorizer, tfidf_matrix, query, inverted_index, k):
    processed_query = remove_stopwords_and_stem(stop_words, query)
    unique_words = set(processed_query)
    if any([w in inverted_index for w in processed_query]):
        relevant_docs = set()
        cos_sim_list = []
        print([processed_query])
        tfidf_query = tfidf_vectorizer.transform([processed_query])
        ready_query = tfidf_query.toarray()[0]
        for query_word in unique_words:
            # Calculating document vectors
            if query_word in inverted_index:
                for doc in inverted_index[query_word][1:]: #first element is the idf
                    relevant_docs.add(doc)
        for doc in relevant_docs:
            cos_sim_list.append([doc, cosine_similarity(ready_query, tfidf_matrix[doc])])
        results = sorted(cos_sim_list, key=lambda item: item[1], reverse=True)[:k]
        return results
    return


def cosine_similarity(query, doc):
    cos_sim = dot(query, doc) / (norm(query) * norm(doc))
    return cos_sim


def main():
    stop_words_list = []
    with open('stopwords.txt', 'r', encoding='utf8') as filestream:
        for line in filestream:
            stop_words_list.append(line.lstrip().rstrip())
    stop_words = set(stop_words_list)

    global speech
    speech = [remove_stopwords_and_stem(stop_words, i) for i in df['speech']]
    df['processed_speech'] = [remove_stopwords_and_stem(stop_words, i) for i in df['speech']]
    #print(speech)
    global names
    names = df['member_name'].unique()

    global parties
    parties = df['political_party'].unique()

    '''
    # Per parliament member
    for name in names:
        print("=======", name, "======")
        temp_list1 = df[df['member_name'] == name]['processed_speech'].tolist()
        speech_of_member = [x for l1 in temp_list1 for x in l1]
        common_words = count_words(speech_of_member)
        #print_weights(common_words, speech_of_member, names)

    # Per parties
    for party in parties:
        print("=======", party, "======")
        temp_list2 = df[df['political_party'] == party]['processed_speech']
        speech_of_party = [x for l2 in temp_list2 for x in l2]
        common_words = count_words(speech_of_party)
        #print_weights(common_words, speech_of_party, parties)
    '''

    # Per speech
    #for doc, s in enumerate(df['processed_speech']):
        #print(doc, s)

    tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda i: i, lowercase=False, smooth_idf=False)  # https://stackoverflow.com/a/31338009
    tfidf_matrix = tfidf_vectorizer.fit_transform(speech)
    tfidf_matrix = tfidf_matrix.toarray()
    tfidf_words = tfidf_vectorizer.get_feature_names_out()

    #df_test = pd.DataFrame(tfidf_matrix, columns=tfidf_words)
    #print(df_test['ΑΠΟΤΥΠΩΘ'])

    #df_result = pd.DataFrame(result.toarray(), columns=tfidf_words)

    inv_index = build_index()
    print(make_query(stop_words, tfidf_vectorizer, tfidf_matrix, "αγροτης στρατος κυβερνηση", inv_index, 10))

    svd = TruncatedSVD(random_state=42)
    U = svd.fit_transform(tfidf_matrix)
    V = svd.components_
    S = svd.singular_values_
    print(U.shape)
    print(S.shape)
    print(V.shape)
    '''
    #U, S, V = np.linalg.svd(tfidf_matrix)
    print(U.shape)
    print(S.shape)
    print(V.shape)
    print(S)

    V_selected = V[:20, :]
    print(V_selected[3])
    M = dot(tfidf_matrix, V_selected.T)
    print(len(tfidf_words))
    '''

if __name__ == "__main__":
    main()
