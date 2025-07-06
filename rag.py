from langchain_huggingface import HuggingFaceEmbeddings # para crear y leer embeddings
from langchain_community.vectorstores import FAISS # la base de datos vectorial a usar es FAISS
from langchain_experimental.text_splitter import SemanticChunker # Para chunkear la informacion
import pandas as pd
import pdfplumber

# extract data

rows = []

with pdfplumber.open('./data/civil-1.pdf') as pdf:
    for page in pdf.pages:
        tables = page.extract_tables()
        for table in tables:
            for row in table[1:]: # ignore header
                if any(str(cell).strip() for cell in row):
                    rows.append(row)

# generate document from data

docs_text = []

for row in rows:
    ciclo, curso, seccion, evento, docente, aula, *dias = row

    dias_texto = ''
    dias_semana = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado"]

    for dia, hora in zip(dias_semana, dias):
        if hora and hora.strip():
            dias_texto += f"Se dicta los {dia.lower()} de {hora.strip()}"

    texto = (
        f"En el {str(ciclo).strip()} se dicta el curso \"{str(curso).strip()}\", "
        f"sección {str(seccion).strip()}, evento {str(evento).strip()}, "
        f"dictado por el docente {str(docente).strip() or 'Desconocido'}, "
        f"ubicado en {str(aula).strip() or 'lugar no especificado'}.{dias_texto}"
    )

    docs_text.append(texto)


data = '\n'.join(docs_text)

# Importando modelo para creacion de embeddings
embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2') # For 

# Dividiendo texto con el metodo SemanticChunker utilizando el embedding importado 
chunker = SemanticChunker(embeddings=embedding, breakpoint_threshold_type="percentile")

# Utilziamos el chunker para dividir la data
documents = chunker.create_documents([data])

# Crear indice FAISS y guardar

vectorstore = FAISS.from_documents(documents, embedding)
vectorstore.save_local("./data/data_horarios")

