import sys
from pathlib import Path
import time
import numpy as np
project_root = Path(__file__).resolve().parent  # carpeta donde está precarga.py
document_path = project_root / "data" / "fashion_products_dataset.json"


from myapp.search.load_corpus import load_corpus
docs = load_corpus(document_path)

from myapp.search.algorithms import create_index_tfidf, compute_line_docs
start_time = time.time()
docs = compute_line_docs(docs)
print("Total time to preprocess documents: {} seconds" .format(np.round(time.time() - start_time, 2)))

start_time = time.time()
num_documents = len(docs)
index, tf, df, idf, title_index, other = create_index_tfidf(docs, num_documents)
print("Total time to create the index: {} seconds" .format(np.round(time.time() - start_time, 2)))

# Ahora guardamos las cosas

final_path = Path("data/index_snapshot.pkl.gz")
import gzip, pickle
with gzip.open(final_path, "wb") as f:
    pickle.dump((index, tf, df, idf, title_index), f, protocol=pickle.HIGHEST_PROTOCOL)

