Introduccion

Para esta primera parte toda la parte de ejecución se realiza en el documento /project_progress/part_1/part_1.ipynb.
La ejecución se realiza ejecutando las celdas del documento part_1.ipynb en orden.

Ejecución

Las primeras celdas son para importar librerias y el json de fashon_products.
Seguidamente se ejecuta una celda para introducir a cada ducumento su "line" correspondiente donde en esta "line" 
está la combinación de strings de todos los campos del documento que son string. Incluídos los del product details.
Aquí es donde se han aplicado las funciones build_terms y compute_line_docs.

Luego, aunque todavia no se tenía que hacer, hemos hecho el buscador tfidf usando la función create_index_tfidf, 
rank documents y search_tfidf, las 2 primeras dadas en las prácticas de la asignatura y adaptadas a nuestra 
estructura de datos.

Por despues de procesar los documentos conseguimos los diccionarios de index tf df title_index e idf.
Luego usamos una query de prueba para probar si el buscador funciona. Al correr el codigo no se tiene
que escribir nada.

Seguidamente hay una seccion de analisis (EDA)
En eta seccion miramos como es nuestro dataset, algunas de sus estadisticas y vemos algunos gráficos.

La ejecución del código deberia ser directa sin complicaciones. El único problema que se podria encontrar es
que no se tengan instaladas alguna de las librerias que usamos. La solucion es usar el comando "pip install libreria_que_falta"


