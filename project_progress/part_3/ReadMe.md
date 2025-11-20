### Reminder:

index[term] ==> [(docid of 1st document that contains this term, [1st position in the document, 2nd position, ...]), 
                        (docid of 2nd document that contains this term, [1st position in the document, 2nd position, ...])]
tf[term] ==> [tf of 1st document that contains this term, tf of 2nd document that contains this term, ...] {dictionary of lists}
df[term] ==> number of documents that contain term
idf[term] ==> log(N/df[term])

queries: lista de strings de las 5 queries
df_queries: lista de todas las queries (las 5) y sus labels en funcion de los documentos
queries_sel: lista de [sub_dataframes de df_queries que solo contienen las rows de la query seleccionada]

# Explicación del Lab 3
Como en los anteriores este notebook puede ser ejecutado directamente (todas las celdas) y así obtendremos:
- Los top documentos de todos los tipos de ranking functions
- La evaluacion de todos los tipos de ranking functions (MAP,MMR,NGDC)

Está todo indicado en el .ipynb
