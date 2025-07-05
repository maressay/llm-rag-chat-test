from langchain_huggingface import HuggingFaceEmbeddings # para crear y leer embeddings
from langchain_community.vectorstores import FAISS # la base de datos vectorial a usar es FAISS
from langchain_experimental.text_splitter import SemanticChunker # Para chunkear la informacion
import pandas as pd

df = pd.read_json('./data/data_horarios_structed.json', encoding='utf-8')
texts = df['texto']
data = '\n'.join(texts.astype(str).to_numpy())

# Importando modelo para creacion de embeddings
embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2') # For 

# Dividiendo texto con el metodo SemanticChunker utilizando el embedding importado 
chunker = SemanticChunker(embeddings=embedding, breakpoint_threshold_type="percentile")

# Utilziamos el chunker para dividir la data
documents = chunker.create_documents([data])

# Crear indice FAISS y guardar

vectorstore = FAISS.from_documents(documents, embedding)
vectorstore.save_local("./data/data_horarios")

