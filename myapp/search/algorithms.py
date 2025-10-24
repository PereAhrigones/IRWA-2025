from nltk.stem import PorterStemmer
import re
from nltk.corpus import stopwords
import collections
from collections import defaultdict
from numpy import linalg as la
import numpy as np

def search_in_corpus(query):
    # 1. create create_tfidf_index

    # 2. apply ranking
    return ""


def build_terms(field):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    line = field.lower()
    line = re.sub(r'([.,])([^\s])', r'\1 \2', line)  # Add space after punctuation if missing
    line = line.split()
    line = [word for word in line if word not in stop_words]
    line = [stemmer.stem(word) for word in line]
    return line

def rank_documents(terms, docs, index, idf, tf, title_index):
    doc_vectors = defaultdict(lambda: [0] * len(terms))
    query_vector = [0] * len(terms)

    query_terms_count = collections.Counter(terms)

    query_norm = la.norm(list(query_terms_count.values()))

    for term_idx, term in enumerate(terms):
        if term not in index:
            continue

        query_vector[term_idx] = (query_terms_count[term] / query_norm) * idf[term]

        for doc_idx, (doc, postings) in enumerate(index[term].items()):
            if doc in docs:
                doc_vectors[doc][term_idx] = tf[term][doc_idx] * idf[term]

    doc_scores = [[np.dot(curDocVec, query_vector), doc] for doc, curDocVec in doc_vectors.items()]
    doc_scores.sort(reverse=True)
    result_docs = [x[1] for x in doc_scores]

    if len(result_docs) == 0:
        print("No results found :(")
        query = input()
        docs = search_tfidf(query, index)

    return result_docs

def search_tfidf(query, index):
    query = build_terms(query)
    docs = set()
    for term in query:
        try:
            term_docs = [posting[0] for posting in index[term]]

            docs.update(term_docs)
        except:
            pass

    docs = list(docs)
    ranked_docs = rank_documents(query, docs, index, idf, tf, title_index)
    return ranked_docs