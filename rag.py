from langchain.embeddings import HuggingFaceEmbeddings # para crear y leer embeddings
from langchain.vectorstores import FAISS # la base de datos vectorial a usar es FAISS
from langchain_experimental.text_splitter import SemanticChunker # Para chunkear la informacion
import pandas as pd

pd.read_json('', encoding='utf-8')
