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
            norm += len(posting[1]) ** 2 # tf_term ^ 2
        norm = math.sqrt(norm) #sqrt(tf_term1 ^ 2 + tf_term2 ^ 2 + ... + tf_termN ^ 2)

        # tf and df weights calculation (dividing the term frequency by the above computed norm) 
        for term, posting in current_page_index.items():
            # append the tf for current term (tf = term frequency in current doc/norm)
            tf[term].append(np.round(len(posting[1])/norm,4)) ##  fij/|dj|

            #increment the document frequency of current term (number of documents containing the current term)
            df[term] += 1 # increment DF for current term

        #merge the current index with the main index
        for term_page, posting_page in current_page_index.items():
            index[term_page].append(posting_page)
            # before index ==> { term: [ [doc1,[1,4,5]], [doc2,[2,3,7]] ] }
            #        current_page_index ==> {term: [current_doc,[1,3,6]] }
            #
            # now    index ==> { term: [ [doc1,[1,4,5]], [doc2,[2,3,7]], [doc3,[1,3,6]] ] }

    # IDF calculation following the formula (3) above
    for term in df:
        idf[term] = np.round(np.log(float(num_documents/df[term])), 4)

    return index, tf, df, idf, title_index

def create_index_tfidf_log(documents, num_documents): # THIS VERSION USES TF IN A LOGARITHMIC SCALE
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

    index_log = defaultdict(list)   # inverted index
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
        # norm is the same for all terms of a document

        # tf and df weights calculation (dividing the term frequency by the above computed norm) 
        norm = 0
        for term, posting in current_page_index.items():
            # posting will contain the list of positions for current term in current document.
            # posting ==> [current_doc, [list of positions]]
            if len(posting[1]) > 0:
                norm += (1 + np.log(len(posting[1]))) ** 2 # tf_term ^ 2
        norm = math.sqrt(norm) #sqrt(tf_term1 ^ 2 + tf_term2 ^ 2 + ... + tf_termN ^ 2)
        
        for term, posting in current_page_index.items():
            if len(posting[1]) > 0:
                tf[term].append((1 + np.log(len(posting[1]))) / norm)

            #increment the document frequency of current term (number of documents containing the current term)
            df[term] += 1 # increment DF for current term

        #merge the current index with the main index
        for term_page, posting_page in current_page_index.items():
            index_log[term_page].append(posting_page)
            # before index ==> { term: [ [doc1,[1,4,5]], [doc2,[2,3,7]] ] }
            #        current_page_index ==> {term: [current_doc,[1,3,6]] }
            #
            # now    index ==> { term: [ [doc1,[1,4,5]], [doc2,[2,3,7]], [doc3,[1,3,6]] ] }

    # IDF calculation following the formula (3) above
    for term in df:
        idf[term] = np.round(np.log(float(num_documents/df[term])), 4)

    return index_log, tf, df, idf, title_index

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

def rank_documents_log(terms, docs, index, idf, tf, title_index): # THIS VERSION USES TF FOR THE QUERY IN A LOGARITHMIC SCALE
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

    doc_norms = {}

    doc_vectors = defaultdict(lambda: [0] * len(terms)) # document vectors initialized to zero
    query_vector = []

    query_terms_count = collections.Counter(terms)  # term frequency in query

    query_norm = la.norm(list(query_terms_count.values())) # norm of query vector

    for term_idx, term in enumerate(terms): # iterate over query terms
        if term not in index: # term not in corpus
            continue

        if term in idf:
            tf_log_q = 1 + np.log(query_terms_count[term])
            query_vector.append(tf_log_q * idf[term])
        else:
            query_vector.append(0.0)
        

        for doc_idx, posting in enumerate(index[term]): # iterate over postings for the term
            try:
                doc_id = posting[0] # get document ID
            except Exception:
                continue
            if doc_id in docs:
                doc_vectors[doc_id][term_idx] = tf[term][doc_idx] * idf[term] # compute document vector component
    
    norm_query = 0
    for w in query_vector:
        norm_query += float(w)**2
    norm_query = math.sqrt(norm_query)
    if norm_query > 0:
        query_vector = [float(w) / norm_query for w in query_vector]

    for doc, curDocVec in doc_vectors.items():
        norm = math.sqrt(sum(float(weight) ** 2 for weight in curDocVec))
        doc_norms[doc] = norm


    doc_scores = [[np.dot(curDocVec, query_vector), doc] for doc, curDocVec in doc_vectors.items()] # compute cosine similarity scores #/(doc_norms[doc]*norm_query)
    doc_scores.sort(reverse=True) # sort documents by score
    result_docs = [x[1] for x in doc_scores] # extract sorted document IDs

    if len(result_docs) == 0:  # no results found
        print("No results found :(")
        query = input()
        docs = search_tfidf(query, index)

    return result_docs

def search_tfidf(query, index, idf, tf, title_index, log = 0):
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
    if log == 1:
        ranked_docs = rank_documents_log(query_terms, docs, index, idf, tf, title_index) # rank documents
    else:
        ranked_docs = rank_documents(query_terms, docs, index, idf, tf, title_index) # rank documents
    return ranked_docs


###### PART 2 ALGORITHMS #####

def select_queries(queries, all_queries): # select only the queries in the dataframe
    """
    queries: list of queries to select
    all_queries: dataframe with all the queries and their labels
    returns: list of dataframes with the selected queries
    """
    queries_selected = [] # list of dataframes
    for q in range(len(queries)):
        #print(q)
        q_temp = all_queries[all_queries['query_id'] == q+1]
        queries_selected.append(q_temp)
    return queries_selected



def precision_at_K(query_selected, ranked_docs, k): # precision at K
    # we suppose that all docs are in query_selected
    """
    query_selected: dataframe with the selected query and its labels
    ranked_docs: list of ranked documents
    k: int
    returns: precision at K
    """
    relevant = []
    cap = len(ranked_docs)
    #print("Total: "+str(cap))

    for r in range(min(k,cap)): # to not exceed the number of ranked documents
        query_row = query_selected[query_selected["pid"] == ranked_docs[r]] # get the row of the document
        l = query_row["labels"] # get the label
        if not l.empty: # if the document has a label
            relevant.append(int(l.iloc[0])) # append the label (1 or 0)
    if relevant: # if there are relevant documents
        return sum(relevant)/k # TP/(TP+FP)
    return 0 # else 0


def recall_at_K(query_selected, ranked_docs, k):
    # we suppose that all docs are in query_selected
    """
    query_selected: dataframe with the selected query and its labels
    ranked_docs: list of ranked documents
    k: int
    returns: recall at K
    """
    relevant = [] # same as precision at K but dividing by total relevant documents
    cap = len(ranked_docs)
    total = query_selected["labels"].sum()

    for r in range(min(k,cap)): # we suppose that if we want to rank a k larger than the ranked documents we still lose precision as we can't find more documents which means not relevant documents
        query_row = query_selected[query_selected["pid"] == ranked_docs[r]]
        l = query_row["labels"]
        if not l.empty:
            relevant.append(int(l.iloc[0]))
    if relevant:
        return sum(relevant)/total # TP/(TP+FN)
    return 0

def average_precision(query_selected, ranked_docs):
    # we suppose that all docs are in query_selected
    """
    query_selected: dataframe with the selected query and its labels
    ranked_docs: list of ranked documents
    returns: average precision
    """
    relevant = 0 # count of relevant documents found
    cap = len(ranked_docs) # total number of ranked documents
    relevance_in_rank = [] # list of precision at each relevant document found

    for r in range(cap):
        query_row = query_selected[query_selected["pid"] == ranked_docs[r]]
        l = query_row["labels"]
        if not l.empty:
            relevant_q = int(l.iloc[0])
            if relevant_q > 0:
                relevant += 1 # count relevant documents found until position r
                relevance_in_rank.append(relevant/(r+1)) # precision at position r
    
    if relevance_in_rank:
        return sum(relevance_in_rank)/len(relevance_in_rank) # average precision
    return 0
    

def f1_score_at_K(query_selected, ranked_docs, k):
    # we suppose that all docs are in query_selected
    """
    query_selected: dataframe with the selected query and its labels
    ranked_docs: list of ranked documents
    k: int
    returns: f1 score at K
    """
    p_p = precision_at_K(query_selected, ranked_docs, k) # precision at K
    r_r = recall_at_K(query_selected, ranked_docs, k) # recall at K

    if p_p * r_r: # to avoid division by zero
        return 2/((1/p_p)+(1/r_r)) # 2*P*R/(P+R), harmonic mean
    return 0

def f1_score_at_K_optimized(query_selected, ranked_docs, k): # optimized version to not calculate twice
    # we suppose that all docs are in query_selected
    """
    query_selected: dataframe with the selected query and its labels
    ranked_docs: list of ranked documents
    k: int
    returns: f1 score at K
    """
    relevant = [] # to store relevant documents found
    cap = len(ranked_docs) # total number of ranked documents
    total = query_selected["labels"].sum() # total number of relevant documents

    for r in range(min(k,cap)):
        query_row = query_selected[query_selected["pid"] == ranked_docs[r]]
        l = query_row["labels"]
        if not l.empty:
            relevant.append(int(l.iloc[0])) # same logic as precision and recall at K
    if relevant:
        return 2*sum(relevant)/(k+total) # simplified formula for F1 score

    return 0

def mean_average_precision(queries_real, queries_test, index, idf, tf, title_index, log = 0, only_id = []):
    """
    query_real: dataframe with all the queries and their labels
    index: inverted index
    idf: inverse document frequency
    tf: term frequency
    title_index: title index
    log: int (1 for log functions, 0 for normal)
    only_id: list of queries to consider instead of all
    returns: mean average precision
    """
    map_map = [] # list of average precisions
    if only_id: # if only is not empty
        queries_id = only_id # only consider these queries
    else:
        queries_id = queries_real["query_id"].astype(int).unique().tolist() # consider all unique queries in the dataframe

    queries = []    
    
    for qid in queries_id:
        queries.append(queries_test[qid-1]) # get the query text from the test dataframe

    queries_sel = select_queries(queries, queries_real) # select the queries from the dataframe

    for i, q in enumerate(queries_sel): # for each selected query
        ranked_docs_t = search_tfidf(queries[i], index, idf, tf, title_index, log) # search for ranked documents
        avgprecision_t = average_precision(q, ranked_docs_t) # compute average precision
        map_map.append(avgprecision_t) # append to list
        # print(avgprecision_t)
    
    if map_map: # if there are average precisions computed
        return sum(map_map)/len(map_map) # return mean average precision
    return 0


def reciprocal_rank(query_sel, ranked_docs, k):
    """
    query_sel: dataframe with the selected query and its labels
    ranked_docs: list of ranked documents
    k: int
    returns: reciprocal rank at K
    """
    cap = len(ranked_docs) 
    # first = 0
    for r in range(min(k,cap)): 
        query_row = query_sel[query_sel["pid"].astype(str) == str(ranked_docs[r])]
        if query_row.empty:
            # No matching row, skip or do nothing
            continue
        else:
            l = query_row["labels"]
            if not l.empty:
                relevant = (int(l.iloc[0]))
                if relevant>0:
                    return(1/(r+1)) # first relevant document found
    
    return 1 / (r + 1)


def mean_reciprocal_rank(queries_real, queries_test, index, idf, tf, title_index, k, log = 0, only_id = []):
    """
    queries_real: dataframe with all the queries and their labels
    queries_test: list of query texts
    index: inverted index
    idf: inverse document frequency
    tf: term frequency
    title_index: title index
    log: int (1 for log functions, 0 for normal)
    only_id: list of queries to consider instead of all
    returns: mean reciprocal rank
    """
    mrr = []
    if only_id: # if only is not empty
        queries_id = only_id # only consider these queries
    else:
        queries_id = queries_real["query_id"].astype(int).unique().tolist() # consider all unique queries in the dataframe

    queries = []
    
    for qid in queries_id:
        queries.append(queries_test[qid-1]) # get the query text from the test dataframe

    queries_sel = select_queries(queries, queries_real)

    for i, q in enumerate(queries_sel):
        ranked_docs_t = search_tfidf(queries[i], index, idf, tf, title_index, log)
        rrt = reciprocal_rank(q, ranked_docs_t, k)
        mrr.append(rrt)
        # print(avgprecision_t)
    
    if mrr:
        return sum(mrr)/len(mrr) # mean reciprocal rank (all same logic as mean average precision)
    return 0

def NDCG(query_selected, ranked_docs, k):
    """
    query_selected: dataframe with the selected query and its labels
    ranked_docs: list of ranked documents
    k: int
    returns: NDCG score at K
    """
    relevant = {} # dictionary to store relevance at each position
    ideal_relevance = [] # list to store ideal relevance
    real_dcg = 0 # real DCG
    ideal_dcg = 0 # ideal DCG
    cap = len(ranked_docs) # total number of ranked documents
    

    for r in range(min(k,cap)): 
        query_row = query_selected[query_selected["pid"] == ranked_docs[r]]
        l = query_row["labels"]
        if not l.empty:
            l = int(l.iloc[0])
            if l > 0:
                relevant[r+1] = l # positions: 1, 2, 3, 4, .... (not 0-indexed but 1-indexed)
    if relevant:
        i = 0
        for position, relevance in relevant.items():
            i+=1 # i starts at 1
            real_dcg += ((2**relevance) - 1)/np.log2(1+position) # formula for DCG
            ideal_relevance.append(relevance) # build ideal relevance list

        ideal_relevance = sorted(ideal_relevance, reverse=True) # sort ideal relevance in descending order
        for i, rel in enumerate(ideal_relevance): 
            if rel > 0:
                ideal_dcg += (2**rel - 1) / np.log2(i + 2) # i starts at 0 here, create ideal DCG
        return real_dcg/ideal_dcg
    
    return 0
            # print(relevance)

            # print("Relevance: "+str(relevance))
            # print("Relevance 2** : "+str(2**relevance))
            # print("Relevance 2** -1 : "+str(2**relevance -1))
            # print("#############")
            # print("Position "+str(position)+ " i="+str(i))
            # print("logp "+str(np.log2(1+position))+ " logi="+str(np.log2(1+i)))

            # print("#############")
