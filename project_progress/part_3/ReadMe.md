Reminder:

index[term] ==> [(docid of 1st document that contains this term, [1st position in the document, 2nd position, ...]), 
                        (docid of 2nd document that contains this term, [1st position in the document, 2nd position, ...])]
tf[term] ==> [tf of 1st document that contains this term, tf of 2nd document that contains this term, ...] {dictionary of lists}
df[term] ==> number of documents that contain term
idf[term] ==> log(N/df[term])