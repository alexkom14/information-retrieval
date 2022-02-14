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
global_names = df['member_name'].unique()
global_parties = df['political_party'].unique()


def remove_stopwords_and_stem(stop_words, speech_one):
    word_tokens = word_tokenize(speech_one)
    filtered_sentence = [w for w in word_tokens if (not w.lower() in stop_words) and w.isalnum()]

    speech_temp = []
    for w in filtered_sentence:
        # https://stackoverflow.com/a/62899722
        d = {ord('\N{COMBINING ACUTE ACCENT}'): None}
        word = unicodedata.normalize('NFD', w).upper().translate(d)
        speech_temp.append(stemmer.stem(word))

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
    column = ''
    if target is global_names:
        column = 'member_name'
    elif target is global_parties:
        column = 'political_party'
    else:
        target = []

    for element in target:
        if any(common_words[i][0] in l for l in df[df[column] == element]['processed_speech']):
            ni += 1

    idf_ = math.log(len(df['processed_speech'])) / ni

    if idf_ < 0.0:
        return 0.0
    return idf_


def idf_simple(word):
    ni = sum(1 for sp in df['processed_speech'] if word in sp)
    idf_ = math.log(len(df['processed_speech'])) / ni

    if idf_ < 0.0:
        return 0.0
    return idf_


def tfidf(i, common_words, s, target):
    return tf(common_words[i][1], s) * idf(i, common_words, target)


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


# k represents top-k results
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
                for doc in inverted_index[query_word][1:]:  # first element is the idf
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
    # import nltk
    # nltk.download('punkt')
    # reading and store stop words
    stop_words_list = []
    with open('stopwords.txt', 'r', encoding='utf8') as filestream:
        for line in filestream:
            stop_words_list.append(line.lstrip().rstrip())
    stop_words = set(stop_words_list)

    # create new column in dataframe. Each row is a processed speech
    df['processed_speech'] = [remove_stopwords_and_stem(stop_words, i) for i in df['speech']]

    # init tfidf vectorizer
    tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda i: i, lowercase=False,
                                       smooth_idf=False)  # https://stackoverflow.com/a/31338009

    # 2d array. columns are each unique word. rows are speeches
    # values are weights of each word
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_speech'])
    tfidf_matrix = tfidf_matrix.toarray()
    tfidf_words = tfidf_vectorizer.get_feature_names_out()

    # df_test = pd.DataFrame(tfidf_matrix, columns=tfidf_words)
    # print(df_test['ΑΠΟΤΥΠΩΘ'])
    # df_result = pd.DataFrame(result.toarray(), columns=tfidf_words)

    """Exercise 1"""
    print("____EXERCISE 1____")
    inv_index = build_index()
    print(inv_index)
    print(make_query(stop_words, tfidf_vectorizer, tfidf_matrix, "αγροτης στρατος κυβερνηση", inv_index, 10))
    """End of Exercise 1"""

    """Exercise 2"""
    print("____EXERCISE 2____")
    # per parliament member
    for name in global_names:
        print("=======", name, "======")
        for year in range(1989, 2021):
            temp_list1 = df[(df['member_name'] == name) & (df['sitting_date'].str.contains(str(year)))]['processed_speech'].tolist()
            speech_of_member = [x for l1 in temp_list1 for x in l1]
            common_words = count_words(speech_of_member)
            print_weights(common_words, speech_of_member, global_names)

    # Per parties
    for party in global_parties:
        print("=======", party, "======")
        for year in range(1989, 2021):
            temp_list2 = df[(df['political_party'] == party) & (df['sitting_date'].str.contains(str(year)))]['processed_speech'].tolist()
            speech_of_party = [x for l2 in temp_list2 for x in l2]
            common_words = count_words(speech_of_party)
            print_weights(common_words, speech_of_party, global_parties)
    """End of Exercise 2"""

    """Exercise 4"""
    print("____EXERCISE 4____")
    svd = TruncatedSVD(random_state=42, n_components=100)
    U = svd.fit_transform(tfidf_matrix)
    V = svd.components_
    S = svd.singular_values_  # ποσο σημαντικο ειναι καθε κονσεπτ

    S_sum = 0
    for i in range(0,len(S)):
        S_sum += S[i]**2

    temp = 0
    threshold = S_sum * 0.8
    flag = 0
    for i in range(0,len(S)):
        temp += S[i]**2
        if temp > threshold:
            flag = i
            break

    V_selected = V[:flag, :]  # καθε σειρα ειναι ενα κονσεπτ και οι τιμες κάθε row δείχνει ποσο συμβάλει στο κονσεπτ
    M = dot(tfidf_matrix, V_selected.T)  # κάθε [ ] είναι ομιλία και έχει N νούμερα κατα πόσο ανήκει στο κάθε κονσεπτ
    print(M)
    """End of Exercise 4"""


if __name__ == "__main__":
    main()
