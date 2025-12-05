# Medical RAG System - Streamlit Application

## Overview

This is a Retrieval-Augmented Generation (RAG) system for clinical diagnostic reasoning using the MIMIC-IV-Ext Direct dataset. The system combines dense (FAISS) and sparse (BM25) retrieval with a generative LLM to provide diagnostic assessments.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you have a CUDA-enabled GPU, you may want to install `faiss-gpu` instead of `faiss-cpu` for better performance:
```bash
pip install faiss-gpu
```

### 2. Verify Directory Structure

Ensure your project has the following structure:
```
RAG_STREAM_LIT/
├── app.py                 # Main Streamlit application
├── config/
│   └── rag_system_config.json
├── models/
│   └── model/
│       ├── model_config.json
│       ├── pytorch_modell/  # Model weights
│       └── Tokenizer/       # Tokenizer files
├── indices/
│   ├── dense_index.faiss
│   ├── bm25_index.pkl
│   ├── corpus_texts.json
│   ├── documents_metadata.json
│   ├── embedder_config.json
│   └── index_stats.json
├── utils/
│   └── rag_loader.py
└── requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

The application will:
1. Load the RAG system (model, tokenizer, indices)
2. Display an interactive interface
3. Allow you to input patient cases and generate diagnostic assessments

## Configuration

System configuration is stored in `config/rag_system_config.json`. The paths are relative to the project root.

### Key Settings:
- **Model paths**: Configured in the config file
- **Retrieval parameters**: Adjustable via the Streamlit sidebar
  - `k`: Number of documents to retrieve (1-10)
  - `alpha`: Balance between dense and sparse retrieval (0.0-1.0)
  - `max_tokens`: Maximum tokens for generation (100-1000)

## Features

- **Hybrid Retrieval**: Combines dense (semantic) and sparse (keyword) retrieval
- **Interactive UI**: User-friendly Streamlit interface
- **Example Cases**: Pre-loaded example patient cases
- **Performance Metrics**: Displays retrieval and generation times
- **Evidence Display**: Shows retrieved documents with relevance scores

## Usage

1. **Load Example Case**: Select from pre-defined examples or enter a custom case
2. **Enter Patient Details**: Describe symptoms, history, vitals, and test results
3. **Generate Diagnosis**: Click the button to retrieve evidence and generate assessment
4. **Review Results**: 
   - View the diagnostic assessment
   - Check performance metrics
   - Examine retrieved evidence documents

## Troubleshooting

### Common Issues:

1. **Model Loading Errors**:
   - Ensure model files are in the correct directory
   - Check that paths in `config/rag_system_config.json` are correct
   - Verify sufficient GPU/CPU memory

2. **Import Errors**:
   - Run `pip install -r requirements.txt`
   - Ensure you're using Python 3.8+

3. **FAISS Index Errors**:
   - Verify `indices/dense_index.faiss` exists
   - Check that the index matches the embedder dimension

4. **BM25 Errors**:
   - Ensure `indices/bm25_index.pkl` exists
   - Verify `rank-bm25` is installed

## Directory Structure

- `app.py` - Main Streamlit application
- `models/` - Saved model and tokenizer
- `indices/` - FAISS and BM25 indices, corpus, and metadata
- `config/` - System configuration
- `utils/` - Utility functions for loading and querying the RAG system
- `data/` - Processed documents (optional)

## Technical Details

- **Retrieval**: Hybrid approach using FAISS (dense) and BM25 (sparse)
- **Embedding Model**: SentenceTransformer (configurable in `embedder_config.json`)
- **Generation Model**: Llama 3.1 8B Instruct (or configured model)
- **Indexing**: FAISS for efficient similarity search

## Important Notes

⚠️ **This is for research/educational purposes only.**
- Always consult healthcare professionals for actual medical decisions
- The system is not intended for clinical use
- Results should be validated by medical experts

## License

This project is for educational purposes as part of the RAG for Diagnostic Reasoning assignment.
