# 📁 Project 1: Multilingual Legal Summarizer (RAG App)
# Structure and Base Code (LangChain + GPT-4 + pgvector)

# Directory structure:
# multilingual-legal-rag/
# ├── app/
# │   ├── main.py               # FastAPI backend
# │   ├── rag_pipeline.py       # RAG pipeline logic
# │   ├── summarizer.py         # GPT-4 summary module
# │   ├── chunker.py            # PDF/text chunking module
# │   ├── embeddings.py         # SentenceTransformer or OpenAI
# │   └── utils.py              # Misc helpers (lang detect, parsing)
# ├── frontend/
# │   ├── streamlit_app.py      # Optional Streamlit UI
# ├── data/
# │   └── example_docs/         # Sample French/English legal PDFs
# ├── models/
# │   └── multilingual_model.py # Custom encoder for embedding
# ├── Dockerfile
# ├── requirements.txt
# └── README.md

# Here is the backend entry point: main.py

from fastapi import FastAPI, UploadFile, File
from app.rag_pipeline import run_rag_pipeline
from app.utils import detect_language

app = FastAPI()

@app.post("/summarize")
async def summarize(file: UploadFile = File(...)):
    contents = await file.read()
    lang = detect_language(contents)
    summary = run_rag_pipeline(file.filename, contents, lang)
    return {"summary": summary, "language": lang}

# utils.py
from langdetect import detect

def detect_language(text_or_bytes):
    try:
        text = text_or_bytes.decode("utf-8") if isinstance(text_or_bytes, bytes) else text_or_bytes
        return detect(text)
    except:
        return "unknown"

# rag_pipeline.py
from app.chunker import chunk_document
from app.embeddings import embed_chunks, retrieve_relevant_chunks
from app.summarizer import generate_summary


def run_rag_pipeline(filename, raw_content, lang):
    chunks = chunk_document(raw_content, lang)
    embedded = embed_chunks(chunks)
    top_chunks = retrieve_relevant_chunks(embedded, lang)
    return generate_summary(top_chunks, lang)

# summarizer.py
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(model_name="gpt-4", temperature=0.3)

prompt_template = PromptTemplate(
    input_variables=["chunks"],
    template="""
    Résumez les informations suivantes en français juridique clair :
    {chunks}
    
    Résumé:
    """
)

def generate_summary(chunks, lang):
    joined = "\n\n".join(chunks)
    prompt = prompt_template.format(chunks=joined)
    return llm.predict(prompt)

# requirements.txt (partial)
fastapi
langchain
openai
pgvector
sentence-transformers
python-magic
uvicorn
streamlit
langdetect
pdfplumber
