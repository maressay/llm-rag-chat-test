from langchain_huggingface import HuggingFaceEmbeddings # para crear y leer embeddings
from langchain_community.vectorstores import FAISS # la base de datos vectorial a usar es FAISS
from langchain_experimental.text_splitter import SemanticChunker # Para chunkear la informacion
import pandas as pd
import pdfplumber

# extract data

all_rows = []

with pdfplumber.open('./data/civil-1.pdf') as pdf:
    for page in pdf.pages:
        tables = page.extract_tables()
        for table in tables:
            df = pd.DataFrame(table[1:], columns=table[0])
            all_rows.append(df)

df_full = pd.concat(all_rows, ignore_index=True)

df_full = df_full.ffill()

# generate document from data

docs_text = []
week_days = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado']

for _, row in df_full.iterrows():

    ciclo = row['Ciclo']
    curso = row['Asignatura']
    seccion = row['Sección']
    evento = row['Evento']
    docente = row['Docente'] or "Desconocido"
    aula = row['Aula'] or "lugar no especificado"

    days_text = ''

    for day in week_days:
        hora = row.get(day, '')
        if pd.notna(hora) and hora.strip():
            days_text += f' Se dicta los {day.lower()} de {hora.strip()}'

    text = (
        f'En el {ciclo.strip()} se dicta el curso \"{curso.strip()}\", '
        f'sección {seccion.strip()}, evento {evento.strip()}, '
        f'dictado por el docente {docente.strip()}, '
        f'ubicado en {aula.strip()}.{days_text}'
    )

    docs_text.append(text)

data = '\n'.join(docs_text)

print(data)

# Importando modelo para creacion de embeddings
embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2') # For 

# Dividiendo texto con el metodo SemanticChunker utilizando el embedding importado 
chunker = SemanticChunker(embeddings=embedding, breakpoint_threshold_type="percentile")

# Utilziamos el chunker para dividir la data
documents = chunker.create_documents([data])

# Crear indice FAISS y guardar

vectorstore = FAISS.from_documents(documents, embedding)
vectorstore.save_local("./data/data_horarios")

