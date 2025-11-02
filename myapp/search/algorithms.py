from nltk.stem import PorterStemmer
import re
from nltk.corpus import stopwords
import collections
from collections import defaultdict
from numpy import linalg as la
import numpy as np

from array import array
import math



long_string_cat = ["title", "description", "brand", "category", "sub_category", "product_details", "seller"] # Fields to be combined for 'line' field (based on our criteria)



def search_in_corpus(query):
    # 1. create create_tfidf_index

    # 2. apply ranking

    #return [pid, title, description, brand, category, sub_category, product_details, seller, out_of_stock,
    # selling_price, discount, actual_price, average_rating, url]
    return ""


def build_terms(field):
    '''
    Function to process a text field into a list of terms as asked in point 1 of part_1
    '''
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    line = field.lower() # Lowercase
    line = re.sub(r'([.,])([^\s])', r'\1 \2', line)  # Add space after punctuation if missing
    line = line.split() # Tokenize
    line = [word for word in line if word not in stop_words] # Remove stopwords
    line = [stemmer.stem(word) for word in line] # Stemming
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

    index = defaultdict(list)   # inverted index
    tf = defaultdict(list)  # term frequencies of terms in documents (documents in the same order as in the main index)
    df = defaultdict(int)  # document frequencies of terms in the corpus
    title_index = defaultdict(str)  # to map document ids to titles
    idf = defaultdict(float) # inverse document frequencies of terms in the corpus

    for doc_id, doc in documents.items():
        doc_id = str(doc_id) # make sure doc_id is string
        terms = doc.line
        title = doc.title
        title_index[doc_id] = title

        current_page_index = {}

        for position, term in enumerate(terms):  ## terms contains all termns in long_string_cat fields of current document
            try:
                # if the term is already in the dict append the position to the corresponding list
                current_page_index[term][1].append(position)
            except:
                # Add the new term as dict key and initialize the array of positions and add the position
                current_page_index[term]=[doc_id, array('I',[position])] #'I' indicates unsigned int (int in Python)

        #normalize term frequencies
        # norm is the same for all terms of a document.

        # norm calculation
        norm = 0
        for term, posting in current_page_index.items():
            # posting will contain the list of positions for current term in current document.
            # posting ==> [current_doc, [list of positions]]
            norm += len(posting[1]) ** 2
        norm = math.sqrt(norm)

        # tf and df weights calculation (dividing the term frequency by the above computed norm) 
        for term, posting in current_page_index.items():
            # append the tf for current term (tf = term frequency in current doc/norm)
            tf[term].append(np.round(len(posting[1])/norm,4)) ## SEE formula (1) above
            #increment the document frequency of current term (number of documents containing the current term)
            df[term] += 1 # increment DF for current term

        #merge the current index with the main index
        for term_page, posting_page in current_page_index.items():
            index[term_page].append(posting_page)

    # IDF calculation following the formula (3) above
    for term in df:
        idf[term] = np.round(np.log(float(num_documents/df[term])), 4)

    return index, tf, df, idf, title_index

def compute_line_docs(documents):
    '''
    Function to compute the 'line' field for each document by combining specified text fields
    Argument:
    documents -- dictionary of Document objects (the corpus)
    Returns:
    documents -- updated dictionary of Document objects with 'line' field computed
    '''
    for doc in documents.values():
        '''
         Combine specified text fields into the 'line' field of Document for search indexing
         '''
        combined_fields = []
        for field_name in long_string_cat: # Fields to be combined for 'line' field
            field_value = getattr(doc, field_name)
            if field_value and isinstance(field_value, str):
                combined_fields.append(field_value)
            elif field_value and isinstance(field_value, dict): # For product_details which is a dict
                combined_fields.append(" ".join(str(v) for v in field_value.values()))
        combined_text = " ".join(combined_fields)
        doc.line = build_terms(combined_text) # Process combined text into terms
    return documents

def rank_documents(terms, docs, index, idf, tf, title_index):
    ''' 
    Rank documents based on cosine similarity between query and document vectors using TF-IDF weights
    Arguments:
    terms -- list of processed query terms
    docs -- list of document IDs to rank
    index -- inverted index
    idf -- inverse document frequencies
    tf -- term frequencies
    title_index -- mapping of document IDs to titles (not needed for ranking for now)
    '''
    doc_vectors = defaultdict(lambda: [0] * len(terms)) # document vectors initialized to zero
    query_vector = [0] * len(terms) # query vector initialized to zero

    query_terms_count = collections.Counter(terms)  # term frequency in query

    query_norm = la.norm(list(query_terms_count.values())) # norm of query vector

    for term_idx, term in enumerate(terms): # iterate over query terms
        if term not in index: # term not in corpus
            continue

        query_vector[term_idx] = (query_terms_count[term] / query_norm) * idf[term] # compute query vector component

        for doc_idx, posting in enumerate(index[term]): # iterate over postings for the term
            try:
                doc_id = posting[0] # get document ID
            except Exception:
                continue
            if doc_id in docs:
                doc_vectors[doc_id][term_idx] = tf[term][doc_idx] * idf[term] # compute document vector component

    doc_scores = [[np.dot(curDocVec, query_vector), doc] for doc, curDocVec in doc_vectors.items()] # compute cosine similarity scores
    doc_scores.sort(reverse=True) # sort documents by score
    result_docs = [x[1] for x in doc_scores] # extract sorted document IDs

    if len(result_docs) == 0:  # no results found
        print("No results found :(")
        query = input()
        docs = search_tfidf(query, index)

    return result_docs

def search_tfidf(query, index, idf, tf, title_index):
    '''Search for documents using the TF-IDF model. Returns only documents that contain ALL query terms.'''
    query_terms = build_terms(query) # process query into terms

    if not query_terms: # If no query terms, return empty list
        return []

    docs_intersection = None
    for term in query_terms: # if term not in index, no document contains it -> no results
        if term not in index:
            return []
        term_docs = {posting[0] for posting in index[term]}
        if docs_intersection is None:
            docs_intersection = term_docs
        else:
            docs_intersection &= term_docs

    if not docs_intersection:
        return []

    docs = list(docs_intersection)
    ranked_docs = rank_documents(query_terms, docs, index, idf, tf, title_index) # rank documents
    return ranked_docs


###### PART 2 ALGORITHMS #####

def select_queries(queries, all_queries):
    queries_selected = []
    for q in queries:
        q_temp = all_queries[all_queries['title'] == q]
        queries_selected.append(q_temp)
    return queries_selected



def precision_at_K(query_selected,ranked_docs, k):
    # we suppose that all docs are in query_selected
    relevant = []
    cap = len(ranked_docs)
    for r in range(min(k,cap)):
        query_row = query_selected[query_selected["pid"] == ranked_docs[r]]
        l = query_row["labels"]
        if not l.empty:
            relevant.append(int(l.iloc[0]))
    if relevant:
        return sum(relevant)/k
    return 0


def recall_at_K():
    # do nothing
    return 0

def average_precision():
    # do nothing
    return 0

def f1_score_at_K():
    # do nothing
    return 0

def mean_average_precision():
    # do nothing
    return 0

def mean_reciprocalr_rank():
    # do nothing
    return 0

def NDCG():
    # do nothing
    return 0

