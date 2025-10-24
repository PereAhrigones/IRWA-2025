from nltk.stem import PorterStemmer
import re
from nltk.corpus import stopwords

from collections import defaultdict
from array import array
import math
import numpy as np
import collections
from numpy import linalg as la

long_string_cat = ["title", "description", "brand", "category", "sub_category", "product_details", "seller"]



def search_in_corpus(query):
    # 1. create create_tfidf_index

    # 2. apply ranking

    #return [pid, title, description, brand, category, sub_category, product_details, seller, out_of_stock,
    # selling_price, discount, actual_price, average_rating, url]
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

def create_index_tfidf(documents, num_documents):
    """
    Implement the inverted index and compute tf, df and idf

    Argument:
    lines -- collection of Wikipedia articles
    num_documents -- total number of documents

    Returns:
    index - the inverted index (implemented through a Python dictionary) containing terms as keys and the corresponding
    list of document these keys appears in (and the positions) as values.
    tf - normalized term frequency for each term in each document
    df - number of documents each term appear in
    idf - inverse document frequency of each term
    """

    index = defaultdict(list)
    tf = defaultdict(list)  # term frequencies of terms in documents (documents in the same order as in the main index)
    df = defaultdict(int)  # document frequencies of terms in the corpus
    title_index = defaultdict(str)
    idf = defaultdict(float)

    for doc_id, doc in documents.items():
        doc_id = int(doc_id)
        terms = doc.line
        title = doc.title
        title_index[doc_id] = title

        ## ===============================================================
        ## create the index for the **current page** and store it in current_page_index
        ## current_page_index ==> { ‘term1’: [current_doc, [list of positions]], ...,‘term_n’: [current_doc, [list of positions]]}

        ## Example: if the curr_doc has id 1 and his text is
        ##"web retrieval information retrieval":

        ## current_page_index ==> { ‘web’: [1, [0]], ‘retrieval’: [1, [1,4]], ‘information’: [1, [2]]}

        ## the term ‘web’ appears in document 1 in positions 0,
        ## the term ‘retrieval’ appears in document 1 in positions 1 and 4
        ## ===============================================================

        current_page_index = {}

        for position, term in enumerate(terms):  ## terms contains page_title + page_text
            try:
                # if the term is already in the dict append the position to the corresponding list
                current_page_index[term][1].append(position)
            except:
                # Add the new term as dict key and initialize the array of positions and add the position
                current_page_index[term]=[doc_id, array('I',[position])] #'I' indicates unsigned int (int in Python)

        #normalize term frequencies
        # Compute the denominator to normalize term frequencies (formula 2 above)
        # norm is the same for all terms of a document.
        norm = 0
        for term, posting in current_page_index.items():
            # posting will contain the list of positions for current term in current document.
            # posting ==> [current_doc, [list of positions]]
            # you can use it to infer the frequency of current term.
            norm += len(posting[1]) ** 2
        norm = math.sqrt(norm)

        # calculate the tf(dividing the term frequency by the above computed norm) and df weights
        for term, posting in current_page_index.items():
            # append the tf for current term (tf = term frequency in current doc/norm)
            tf[term].append(np.round(len(posting[1])/norm,4)) ## SEE formula (1) above
            #increment the document frequency of current term (number of documents containing the current term)
            df[term] += 1 # increment DF for current term

        #merge the current page index with the main index
        for term_page, posting_page in current_page_index.items():
            index[term_page].append(posting_page)

    # Compute IDF following the formula (3) above. HINT: use np.log
    # Note: It is computed later after we know the df.
    for term in df:
        idf[term] = np.round(np.log(float(num_documents/df[term])), 4)

    return index, tf, df, idf, title_index

def compute_line_docs(documents):
    for doc in documents.values():
        combined_fields = []
        for field_name in long_string_cat:
            field_value = getattr(doc, field_name)
            if field_value:
                combined_fields.append(field_value)
        combined_text = " ".join(combined_fields)
        doc.line = build_terms(combined_text)
    return documents

