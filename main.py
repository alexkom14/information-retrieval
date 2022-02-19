from nltk.tokenize import word_tokenize
from greek_stemmer import GreekStemmer  # https://github.com/alup/python_greek_stemmer
import pandas as pd
import unicodedata
# from pandas.core.base import SpecificationError
import numpy as np
from numpy import dot
from numpy.linalg import norm
from collections import Counter
# import math
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from random import shuffle
from operator import itemgetter

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


# # N represents how many words to be displayed
# def print_weights(common_words, s, target, N):
#     for i in range(min(len(common_words), N)):
#         print(common_words[i][0], "TF-IDF:", tfidf(i, common_words, s, target))


# def tf(word_count, s):
#     return word_count / len(s)


# # PARTY AND MEMBER
# def idf(i, common_words, target):
#     ni = 0
#     column = ''
#     if target is global_names:
#         column = 'member_name'
#     elif target is global_parties:
#         column = 'political_party'
#     else:
#         target = []
#
#     for element in target:
#         if any(common_words[i][0] in l for l in df[df[column] == element]['processed_speech']):
#             ni += 1
#
#     idf_ = math.log(len(df['processed_speech'])) / ni
#
#     if idf_ < 0.0:
#         return 0.0
#     return idf_


# def idf_simple(word):
#     ni = sum(1 for sp in df['processed_speech'] if word in sp)
#     idf_ = math.log(len(df['processed_speech'])) / ni
#
#     if idf_ < 0.0:
#         return 0.0
#     return idf_


# def tfidf(i, common_words, s, target):
#     return tf(common_words[i][1], s) * idf(i, common_words, target)


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


def create_hash_f(size):
    hash_ex = list(range(1, size+1))
    shuffle(hash_ex)
    return hash_ex


def build_minhash_func(vocab_size, nbits):
    hashes = []
    for i in range(nbits):
        hashes.append(create_hash_f(vocab_size))
    return hashes


def create_hash(vector, minhash_func, size):
    signature = []
    for func in minhash_func:
        for i in range(1, size + 1):
            idx = func.index(i)
            signature_val = vector[idx]
            if signature_val == 1:
                signature.append(i)
                break
    return signature


def jaccard(x, y):
    return len(x.intersection(y)) / len(x.union(y))


def split_vector(signature, b):
    assert  len(signature) % b == 0
    r = int(len(signature) / b)
    subvecs = []
    for i in range(0, len(signature), r):
        subvecs.append(signature[i:i+r])
    return subvecs


def main():
    # import nltk
    # nltk.download('punkt')
    # reading and store stop words
    K = 3  # variable to set top - k
    stop_words_list = []
    with open('stopwords.txt', 'r', encoding='utf8') as filestream:
        for line in filestream:
            stop_words_list.append(line.lstrip().rstrip())
    stop_words = set(stop_words_list)

    # create new column in dataframe. Each row is a processed speech
    df['processed_speech'] = [remove_stopwords_and_stem(stop_words, i) for i in df['speech']]
    df['year'] = [df['sitting_date'][y][-4:] for y in range(df.shape[0])]  # new column year

    ### MAYBE DELETE SPEECH AND SITTING_DATE COLUMN

    # init tfidf vectorizer
    tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda i: i, lowercase=False,
                                       smooth_idf=False)  # https://stackoverflow.com/a/31338009

    # 2d array. columns are each unique word. rows are speeches
    # values are weights of each word
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_speech'])
    tfidf_matrix = tfidf_matrix.toarray()

    feature_array = np.array(tfidf_vectorizer.get_feature_names_out()) # array containing every word of the speeches

    #df_test = pd.DataFrame(tfidf_matrix, columns=tfidf_words)
    #print(df_test)
    #print("test", df_test['ΕΥΧΑΡΙΣΤ'])
    #df_result = pd.DataFrame(result.toarray(), columns=tfidf_words)

    # """Exercise 1"""
    # print("____EXERCISE 1____")
    # inv_index = build_index()
    # print(inv_index)
    # print(make_query(stop_words, tfidf_vectorizer, tfidf_matrix, "αγροτης στρατος κυβερνηση", inv_index, 10))
    # """End of Exercise 1"""

    # """Exercise 2"""
    # print("____EXERCISE 2____")
    # print()
    # # per speech
    # # calculates top k of one speech
    # for i, s in enumerate(df['processed_speech']):
    #     print("=======Speech", i+1, "======")
    #     top_indexes = np.argpartition(tfidf_matrix[i], -K)[-K:]  # finds the indexes where the top k values are located
    #     for j in top_indexes:
    #         print("word: ", feature_array[j], "tfidf: ", tfidf_matrix[i][j])
    # print()
    # # per parliament member
    # for year in range(1989, 2021):
    #     speech_per_member = df[df.year == str(year)].groupby(['member_name'], as_index=False).agg({'processed_speech': 'sum'})  # group by member and concat the processed_speech cells
    #     if not speech_per_member.empty:
    #         print("=======", year, "========")
    #         member_tfidf = tfidf_vectorizer.fit_transform(speech_per_member['processed_speech']).toarray()  # calculate tf-idf
    #         member_feature_array = np.array(tfidf_vectorizer.get_feature_names_out())  # array containing every word of the speeches
    #         for i, name in enumerate(speech_per_member['member_name']):
    #             top_indexes = np.argpartition(member_tfidf[i], -K)[-K:]  # finds the indexes where the top k values are located
    #             print("------", name, "------")
    #             for j in top_indexes:
    #                 print("word: ", member_feature_array[j], "tfidf: ", member_tfidf[i][j])
    # print()
    # # Per party
    # for year in range(1989, 2021):
    #     speech_per_party = df[df.year == str(year)].groupby(['political_party'], as_index=False).agg({'processed_speech': 'sum'}) #group by party and concat the processed_speech cells
    #     if not speech_per_party.empty:
    #         print("=======", year, "========")
    #         party_tfidf = tfidf_vectorizer.fit_transform(speech_per_party['processed_speech']).toarray() # calculate tf-idf
    #         party_feature_array = np.array(tfidf_vectorizer.get_feature_names_out()) # array containing every word of the speeches
    #         for i, party_name in enumerate(speech_per_party['political_party']):
    #             top_indexes = np.argpartition(party_tfidf[i], -K)[-K:]  # finds the indexes where the top k values are located
    #             print("------", party_name, "------")
    #             for j in top_indexes:
    #                 print("word: ", party_feature_array[j], "tfidf: ", party_tfidf[i][j])
    # """End of Exercise 2"""

    """Exercise 3"""
    speech_per_member = df[df.year == str(2020)].groupby(['member_name'], as_index=False).agg({'processed_speech': 'sum'})
    speech_per_member['processed_speech'] = speech_per_member['processed_speech'].apply(set)

    member_speech_list = []
    for i in range(0, len(speech_per_member['processed_speech'])):
        member_speech_list.append(speech_per_member.at[i, 'processed_speech'])
    vocab = set().union(*member_speech_list)

    hot_list = []
    for i in range(0, len(speech_per_member['processed_speech'])):
        a = member_speech_list[i]
        hot_list.append([1 if x in a else 0 for x in vocab])

    hash_ex = list(range(1, len(vocab)+1))
    shuffle(hash_ex)
    # create 20 minhash vectors
    minhash_func = build_minhash_func(len(vocab), 20)
    sig_list = []
    for i in range(0, len(speech_per_member['processed_speech'])):
        sig_list.append((create_hash(hot_list[i], minhash_func, len(vocab))))

    # for i in range(0, len(speech_per_member['processed_speech'])):
    #     for j in range(i+1, len(speech_per_member['processed_speech'])):
    #         print(jaccard(sig_list[i], sig_list[j]))

    band_list = []
    for i in range(0, len(sig_list)):
        band_list.append(split_vector(sig_list[i], 10))

    candidate_pairs = []
    for i in range(0, len(band_list)):
        for j in range(i+1, len(band_list)):
            for i_rows, j_rows in zip(band_list[i], band_list[j]):
                if i_rows == j_rows:
                    candidate_pairs.append([i, j])
                    break

    similarities = []
    for i in range(0, len(candidate_pairs)):
        candidates = candidate_pairs[i]
        can1 = candidates[0]
        can2 = candidates[1]
        sim = jaccard(set(sig_list[can1]), set(sig_list[can2]))
        similarities.append([sim, can1, can2])

    sorted_similarities = sorted(similarities, key=itemgetter(0), reverse=True)
    k = int(input("Give an integer k for top-k similar member couples, depend on their speeches: "))

    if k > len(sorted_similarities):
        k = len(sorted_similarities)
    for i in range(0, k):
        sort_sim = sorted_similarities[i]
        sim = sort_sim[0]
        mem1 = df['member_name'][sort_sim[1]]
        mem2 = df['member_name'][sort_sim[2]]
        print(f"Couple {mem1} and {mem2} has {sim} similarity, depend on their speeches.")
    """End of Exercise 3"""

    # """Exercise 4"""
    # print("____EXERCISE 4____")
    # svd = TruncatedSVD(random_state=42, n_components=100)
    # U = svd.fit_transform(tfidf_matrix)
    # V = svd.components_
    # S = svd.singular_values_  # ποσο σημαντικο ειναι καθε κονσεπτ
    #
    # S_sum = 0
    # for i in range(0,len(S)):
    #     S_sum += S[i]**2
    #
    # temp = 0
    # threshold = S_sum * 0.8
    # flag = 0
    # for i in range(0,len(S)):
    #     temp += S[i]**2
    #     if temp > threshold:
    #         flag = i
    #         break
    #
    # V_selected = V[:flag, :]  # καθε σειρα ειναι ενα κονσεπτ και οι τιμες κάθε row δείχνει ποσο συμβάλει στο κονσεπτ
    # M = dot(tfidf_matrix, V_selected.T)  # κάθε [ ] είναι ομιλία και έχει N νούμερα κατα πόσο ανήκει στο κάθε κονσεπτ
    # print(M)
    # """End of Exercise 4"""


if __name__ == "__main__":
    main()
