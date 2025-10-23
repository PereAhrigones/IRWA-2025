from nltk.stem import PorterStemmer
import re
from nltk.corpus import stopwords

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