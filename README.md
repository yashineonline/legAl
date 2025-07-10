# 📁 Project 1: Multilingual Legal Summarizer (RAG App)
# Now includes: French benchmark set, BLOOMZ prompt-tuning, HuggingFace Spaces demo, GitHub CI/CD, README automation

# Directory structure:
# multilingual-legal-rag/
# ├── .github/
# │   └── workflows/
# │       └── deploy.yml          # CI/CD GitHub Actions pipeline
# ├── app/
# │   ├── main.py                 # FastAPI backend
# │   ├── rag_pipeline.py         # RAG pipeline logic
# │   ├── summarizer.py           # GPT-4 summary module
# │   ├── chunker.py              # PDF/text chunking module
# │   ├── embeddings.py           # BLOOMZ/M-BERT fine-tuning support
# │   ├── citations.py            # Extract citations and references
# │   └── utils.py                # Lang detect, token validation, benchmarks
# ├── benchmark/
# │   ├── raw_pdfs/               # French legal PDF corpus
# │   ├── expected_outputs.json   # GPT-reviewed summaries
# │   └── metrics.py              # Rouge/BLEU scoring tools
# ├── training/
# │   ├── dataset_builder.py      # Prepares HF Dataset format
# │   ├── train_bloomz.py         # Prompt tuning loop using PEFT
# │   └── eval_bloomz.py          # Eval script on benchmark set
# ├── frontend/
# │   ├── streamlit_app.py        # Optional Streamlit UI
# ├── models/
# │   └── multilingual_model.py   # Wrapper for tuned BLOOMZ embeddings
# ├── .huggingface/               # HF Spaces config
# │   └── README.md               # Auto-generated README for demo card
# ├── Dockerfile
# ├── requirements.txt
# ├── README.md                   # GitHub landing page (auto-filled)

# ✅ README.md (GitHub view)
"""
# 🧾 Multilingual Legal Document Summarizer (RAG App)

A multilingual GenAI tool for summarizing and retrieving information from legal documents, especially in French. Powered by LangChain, BLOOMZ, FastAPI, and Streamlit.

### 🌍 Features
- 📚 Upload French/English PDFs
- 🧠 Summarize with GPT-4 or tuned BLOOMZ
- 🔍 Semantic retrieval with embeddings
- 🧾 Auto-generate citation graphs
- 🚀 Fully containerized, deployable to HuggingFace Spaces

### 🚦 Quickstart
```bash
git clone https://github.com/YOUR_USERNAME/multilingual-legal-rag
cd multilingual-legal-rag
pip install -r requirements.txt
streamlit run frontend/streamlit_app.py
```

### 🧪 Benchmarking
Run:
```bash
python benchmark/metrics.py
```
Compare output summaries to manually curated ones in `benchmark/expected_outputs.json`

### 🛠 Fine-tune BLOOMZ (Optional)
```bash
python training/train_bloomz.py --model bigscience/bloomz-560m --dataset benchmark/expected_outputs.json
```

### 🤝 License
MIT License — see [LICENSE](./LICENSE)

### 👤 Author
[Yashine Hazmatally Goolam Hossen](mailto:yashineonline@gmail.com)
"""

# ✅ .huggingface/README.md (for HuggingFace Spaces Card)
"""
---
title: "Multilingual Legal RAG Summarizer"
tags:
  - rag
  - french
  - legal-ai
  - multilingual
  - huggingface-spaces
  - bloomz
app_file: frontend/streamlit_app.py
sdk: streamlit
license: mit
---

# 📄 Multilingual Legal Summarizer

Summarize French legal documents using GPT-4 or BLOOMZ. Uses RAG pipelines, semantic chunking, reranking, and document-aware summarization. Upload your own PDF and test it live.
"""

# All now auto-included and deploy-ready.





