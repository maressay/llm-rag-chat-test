from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.utils.quantization_config import BitsAndBytesConfig
import torch

# imports for RAG
from langchain_huggingface import HuggingFaceEmbeddings # para crear y leer embeddings
from langchain_community.vectorstores import FAISS # la base de datos vectorial a usar es FAISS

# load embeddings
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("./data/data_horarios", embeddings = embedding, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(sarch_kwargs={"k": 3})

# load model
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model_id = "microsoft/Phi-3-mini-128k-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    quantization_config=quant_config,
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, max_length=500)

pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, return_full_text=False)

# input user

input_user = input('You 🥸: ')

# get context from FAISS

docs = retriever.invoke(input_user)
context = "\n".join([doc.page_content for doc in docs])

# building prompt

prompt = f"""
<|system|>
Eres un asistente académico que responde preguntas sobre los horarios de clases de los alumnos.
Tu única fuente de información es el contexto proporcionado. No inventes respuestas. Si no sabes algo, responde educadamente que no puedes ayudar.
<|end|>
<|user|>
Contexto:

{context}

Pregunta del usuario: {input_user}
<|end|>
<|assistant|>
"""

# generate answer

output = pipe(
    prompt,
    max_new_tokens=300,
    temperature=0.1,
    do_sample=False,
    return_full_text=False
    )

print('Model 🤖: ', output[0]["generated_text"])

