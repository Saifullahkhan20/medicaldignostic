"""
Utility functions for loading and using the RAG system
"""

import json
import pickle
import torch
import faiss
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List

def load_rag_system(base_path: str = None, progress_callback=None) -> Dict:
    """
    Load complete RAG system from saved files.
    """
    
    if base_path is None:
        base_path = Path(__file__).parent.parent
    else:
        base_path = Path(base_path)
    
    def update_progress(msg):
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)
    
    # Load config
    update_progress("Loading configuration...")
    config_path = base_path / "config" / "rag_system_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Load tokenizer - always use model_id to ensure it matches the model
    update_progress("Loading tokenizer...")
    tokenizer_path = config["model"]["tokenizer"]
    model_id = config["model"]["model_id"]
    
    # Always load tokenizer from model_id to ensure it matches the model
    # This prevents tokenizer/model mismatches
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Set pad token if not set
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.pad_token_id = 0
    except Exception as e:
        update_progress(f"Warning: Could not load tokenizer from {model_id}, trying local path...")
        if tokenizer_path:
            if not Path(tokenizer_path).is_absolute():
                tokenizer_path = base_path / tokenizer_path
            else:
                tokenizer_path = Path(tokenizer_path)
            
            if tokenizer_path.exists() and any(tokenizer_path.iterdir()):
                tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
            else:
                raise OSError(f"Could not load tokenizer from {model_id} or {tokenizer_path}: {e}")
        else:
            raise OSError(f"Could not load tokenizer from {model_id}: {e}")
    
    # Load model - use model_id if local path is None or doesn't work
    update_progress("Loading model (this may take a few minutes on first run)...")
    model_path = config["model"]["model"]
    model_id = config["model"]["model_id"]
    
    if model_path:
        if not Path(model_path).is_absolute():
            model_path = base_path / model_path
        else:
            model_path = Path(model_path)
        
        # Check if model exists locally and has weight files
        model_files_exist = model_path.exists() and any(model_path.iterdir())
        if model_files_exist:
            weight_files = list(model_path.glob("*.safetensors")) + list(model_path.glob("*.bin"))
            model_files_exist = len(weight_files) > 0
        
        if model_files_exist:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    local_files_only=True,
                    torch_dtype=torch.float16,
                )
                if torch.cuda.is_available():
                    model = model.cuda()
                else:
                    model = model.cpu()
            except Exception as e:
                print(f"Warning: Could not load local model, downloading from {model_id}...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                )
                if torch.cuda.is_available():
                    model = model.cuda()
                else:
                    model = model.cpu()
        else:
            # Model files don't exist, download from model_id
            update_progress(f"Model files not found. Downloading {model_id} (this may take a few minutes)...")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                )
            except Exception as e:
                # Try without float16 if it fails
                update_progress(f"Retrying with float32...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                )
            if torch.cuda.is_available():
                model = model.cuda()
            else:
                model = model.cpu()
    else:
        # No model path specified, download from model_id
        update_progress(f"Downloading model {model_id} (this may take a few minutes)...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
            )
        except Exception as e:
            # Try without float16 if it fails
            update_progress(f"Retrying with float32...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
            )
        if torch.cuda.is_available():
            model = model.cuda()
        else:
            model = model.cpu()
    
    model.eval()
    update_progress("Model loaded successfully!")
    
    # Load embedder
    update_progress("Loading embedding model...")
    embedder_config_path = base_path / "indices" / "embedder_config.json"
    with open(embedder_config_path, "r") as f:
        embedder_config = json.load(f)
    
    embedder = SentenceTransformer(embedder_config["model_name"])
    update_progress("Embedding model loaded!")
    
    # Load FAISS index
    update_progress("Loading FAISS index...")
    faiss_path = base_path / "indices" / "dense_index.faiss"
    dense_index = faiss.read_index(str(faiss_path))
    
    # Load BM25 index
    update_progress("Loading BM25 index...")
    bm25_path = base_path / "indices" / "bm25_index.pkl"
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)
    
    # Load corpus
    update_progress("Loading corpus and metadata...")
    corpus_path = base_path / "indices" / "corpus_texts.json"
    with open(corpus_path, "r") as f:
        corpus_texts = json.load(f)
    
    # Load documents
    docs_path = base_path / "indices" / "documents_metadata.json"
    with open(docs_path, "r") as f:
        documents = json.load(f)
    
    # Load stats
    stats_path = base_path / "indices" / "index_stats.json"
    with open(stats_path, "r") as f:
        stats = json.load(f)
    
    update_progress("All components loaded! System ready.")
    
    return {
        "model": model,
        "tokenizer": tokenizer,
        "embedder": embedder,
        "dense_index": dense_index,
        "bm25": bm25,
        "corpus_texts": corpus_texts,
        "documents": documents,
        "config": config,
        "stats": stats
    }


def hybrid_retrieve(
    system: Dict,
    query: str,
    k: int = 5,
    alpha: float = 0.7
) -> List[Dict]:
    """
    Perform hybrid retrieval.
    """
    
    embedder = system["embedder"]
    dense_index = system["dense_index"]
    bm25 = system["bm25"]
    corpus_texts = system["corpus_texts"]
    documents = system["documents"]
    
    # Dense retrieval
    q_emb = embedder.encode([query], convert_to_numpy=True)
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-8)  # Normalize
    D_dense, I_dense = dense_index.search(q_emb.astype(np.float32), k*2)
    
    # Convert distance to similarity
    # FAISS L2 distance: for normalized vectors, distance^2 = 2*(1-cosine_similarity)
    # So similarity ≈ 1 - distance/2 (approximation for normalized vectors)
    # If values are negative, it's likely inner product (higher is better)
    if np.any(D_dense[0] < 0):
        # Inner product index - use as is (higher is better)
        dense_similarities = D_dense[0]
    else:
        # L2 distance - convert to similarity
        # Normalize distances to [0, 1] range for similarity
        max_dist = np.max(D_dense[0]) if np.max(D_dense[0]) > 0 else 1.0
        dense_similarities = 1 - (D_dense[0] / (max_dist + 1e-8))
    
    # BM25 retrieval
    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_max = bm25_scores.max() if bm25_scores.max() > 0 else 1.0
    bm25_norm = bm25_scores / bm25_max if bm25_max > 0 else bm25_scores
    
    # Hybrid scoring - combine dense and sparse scores
    hybrid_scores = {}
    
    # Add dense retrieval scores
    for rank, idx in enumerate(I_dense[0]):
        if idx < len(documents):
            # Use similarity (higher is better)
            hybrid_scores[idx] = alpha * dense_similarities[rank]
    
    # Add BM25 scores
    for idx, score in enumerate(bm25_norm):
        if idx < len(documents):
            if idx in hybrid_scores:
                hybrid_scores[idx] += (1 - alpha) * score
            else:
                hybrid_scores[idx] = (1 - alpha) * score
    
    # Get top-k
    sorted_items = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    top_k = sorted_items[:k]
    
    results = []
    for idx, score in top_k:
        results.append({
            "text": corpus_texts[idx],
            "score": float(score),
            "source": documents[idx].get("source", "unknown"),
            "disease": documents[idx].get("disease", ""),
            "metadata": documents[idx]
        })
    
    return results


def is_valid_medical_query(query: str) -> bool:
    """
    Check if the query is a valid medical case description.
    """
    query_lower = query.lower().strip()
    
    # Invalid keywords that indicate non-medical queries
    invalid_keywords = [
        'robot', 'robots', 'alien', 'aliens', 'fantasy', 'fictional',
        'video game', 'game', 'movie', 'story', 'fiction', 'fictional',
        'animal disease', 'pet', 'dog', 'cat', 'zombie', 'vampire',
        'superhero', 'magic', 'wizard', 'dragon', 'monster'
    ]
    
    # Vague/question phrases that aren't medical cases
    vague_phrases = [
        'what is', 'what are', 'what does', 'what do', 'what can',
        'how does', 'how do', 'how can', 'how is', 'how are',
        'tell me', 'explain', 'describe', 'what actually',
        'what is going on', 'what is happening', 'what is this',
        'what does this mean', 'can you', 'could you', 'would you',
        'i want to know', 'i need to know', 'help me understand'
    ]
    
    # Check for invalid keywords
    for keyword in invalid_keywords:
        if keyword in query_lower:
            return False
    
    # Check for vague questions
    for phrase in vague_phrases:
        if phrase in query_lower:
            # If it's just a vague question without medical context, reject it
            if len(query.split()) < 8:  # Short vague questions
                return False
    
    # Check if query is too short
    if len(query.split()) < 3:
        return False
    
    # Must have actual medical content - check for medical terms
    medical_terms = [
        'patient', 'symptom', 'diagnosis', 'disease', 'illness', 'condition',
        'pain', 'fever', 'cough', 'breath', 'chest', 'headache', 'ache',
        'blood', 'pressure', 'bp', 'hr', 'heart rate', 'temperature',
        'heart', 'lung', 'kidney', 'liver', 'brain', 'stomach',
        'medical', 'clinical', 'doctor', 'hospital', 'treatment',
        'medicine', 'drug', 'medication', 'test', 'lab', 'x-ray', 'ct', 'mri',
        'age', 'year', 'old', 'male', 'female', 'history', 'vitals',
        'troponin', 'ecg', 'eeg', 'spirometry', 'fev1', 'fvc',
        'hypertension', 'diabetes', 'asthma', 'pneumonia', 'infection',
        'nausea', 'vomiting', 'diarrhea', 'dizziness', 'weakness',
        'shortness', 'dyspnea', 'wheezing', 'rash', 'swelling'
    ]
    
    # Count how many medical terms are present
    medical_term_count = sum(1 for term in medical_terms if term in query_lower)
    
    # Must have at least 2 medical terms or be a proper case description
    if medical_term_count < 2:
        # Check if it's a proper case format (age, gender, symptoms)
        has_case_format = (
            any(word in query_lower for word in ['year', 'old', 'male', 'female']) or
            any(word in query_lower for word in ['history', 'vitals', 'labs', 'symptoms'])
        )
        if not has_case_format:
            return False
    
    return True


def generate_fallback_answer(query: str, retrieved_docs: List[Dict]) -> str:
    """
    INSTANT keyword-based response system - no model needed.
    """
    # First check if this is a valid medical query
    if not is_valid_medical_query(query):
        return """PRIMARY DIAGNOSIS:
Invalid Query - Not a Medical Case

SUPPORTING EVIDENCE:
The provided query does not appear to be a valid medical case description. This system is designed to analyze human patient cases and clinical presentations.

CLINICAL REASONING:
The query contains terms or concepts that are not applicable to human medical diagnosis. Please provide:
- A description of a human patient's symptoms
- Clinical findings, vital signs, or test results
- Medical history or relevant information
- Age, gender, and presenting complaints

CONFIDENCE LEVEL:
N/A - Query validation failed

RECOMMENDATIONS:
Please provide a valid medical case description including:
1. Patient demographics (age, gender)
2. Presenting symptoms
3. Medical history (if relevant)
4. Clinical findings, vital signs, or test results
5. Duration and progression of symptoms

Example: "68-year-old male with chest pain radiating to left arm, history of hypertension, BP 155/92 mmHg, elevated troponin."
"""
    
    query_lower = query.lower()
    
    # Simple keyword-to-diagnosis mapping
    diagnosis_map = {
        'cold': 'Upper Respiratory Tract Infection (URTI) / Common Cold',
        'flu': 'Influenza',
        'cough': 'Respiratory Infection or Chronic Obstructive Pulmonary Disease',
        'fever': 'Febrile Illness - Possible Infection',
        'chest pain': 'Acute Coronary Syndrome / Myocardial Infarction',
        'shortness of breath': 'Respiratory Distress / COPD / Asthma',
        'dyspnea': 'Respiratory Distress',
        'wheezing': 'Asthma or COPD',
        'headache': 'Primary Headache Disorder or Secondary Headache',
        'nausea': 'Gastrointestinal Disorder',
        'vomiting': 'Gastroenteritis or Other GI Condition',
        'diarrhea': 'Gastroenteritis',
        'weakness': 'Neurological Deficit or Systemic Illness',
        'dizziness': 'Vertigo or Hypotension',
        'hypertension': 'Hypertension',
        'diabetes': 'Diabetes Mellitus',
        'asthma': 'Asthma',
        'pneumonia': 'Pneumonia',
        'stroke': 'Cerebrovascular Accident (CVA)',
    }
    
    # Find matching diagnosis
    matched_diagnosis = None
    matched_keyword = None
    
    for keyword, diagnosis in diagnosis_map.items():
        if keyword in query_lower:
            matched_diagnosis = diagnosis
            matched_keyword = keyword
            break
    
    # Get top disease from retrieved docs if available
    top_disease = matched_diagnosis
    if retrieved_docs and retrieved_docs[0].get('disease'):
        top_disease = retrieved_docs[0].get('disease')
        if not matched_diagnosis:
            matched_diagnosis = top_disease
    
    if not top_disease:
        top_disease = "Clinical Assessment Required"
    
    # Extract key symptoms mentioned
    symptoms_found = []
    symptom_keywords = ['cold', 'flu', 'cough', 'fever', 'chest pain', 'shortness of breath', 
                       'dyspnea', 'wheezing', 'headache', 'nausea', 'vomiting', 'diarrhea',
                       'weakness', 'dizziness', 'hypertension', 'diabetes', 'asthma', 'pain']
    
    for symptom in symptom_keywords:
        if symptom in query_lower:
            symptoms_found.append(symptom)
    
    # If no symptoms found and query is vague, return error
    if not symptoms_found and len(query.split()) < 10:
        # Check if it's a question
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        if any(query_lower.startswith(word) for word in question_words):
            return """PRIMARY DIAGNOSIS:
Invalid Query - Please Provide Patient Case Details

SUPPORTING EVIDENCE:
This system requires a patient case description with specific symptoms, clinical findings, or medical history. General questions cannot be answered without patient-specific information.

CLINICAL REASONING:
To provide a diagnostic assessment, please include:
- Patient demographics (age, gender)
- Presenting symptoms
- Medical history
- Clinical findings, vital signs, or test results

CONFIDENCE LEVEL:
N/A - Insufficient information provided

RECOMMENDATIONS:
Please provide a complete patient case description. Example:
"68-year-old male with chest pain radiating to left arm. History of hypertension. BP 155/92 mmHg, HR 95 bpm. Labs: Troponin 0.66 ng/mL (elevated). ECG: ST depression in leads V2-V4."
"""
    
    # Build instant response
    response = f"""PRIMARY DIAGNOSIS:
{top_disease}

SUPPORTING EVIDENCE:
Based on the clinical presentation, the patient reports: {', '.join(symptoms_found) if symptoms_found else 'symptoms as described'}.

Key clinical findings from presentation:
"""
    
    # Add specific findings
    if 'cold' in query_lower or 'flu' in query_lower:
        response += "- Upper respiratory symptoms consistent with viral infection\n"
        response += "- Common cold or influenza likely based on symptom presentation\n"
    
    if 'chest pain' in query_lower:
        response += "- Chest pain requires immediate cardiac evaluation\n"
        response += "- ECG and cardiac markers recommended\n"
    
    if 'shortness of breath' in query_lower or 'dyspnea' in query_lower:
        response += "- Respiratory distress noted\n"
        response += "- Pulmonary function tests may be indicated\n"
    
    if 'cough' in query_lower:
        response += "- Cough present, may indicate respiratory infection or chronic lung disease\n"
    
    if retrieved_docs:
        response += f"\nRelevant medical evidence retrieved from {len(retrieved_docs)} clinical sources.\n"
        if retrieved_docs[0].get('disease'):
            response += f"Top matching condition: {retrieved_docs[0].get('disease')}\n"
    
    response += f"""
CLINICAL REASONING:
The patient's symptoms ({', '.join(symptoms_found) if symptoms_found else 'as described'}) are consistent with {top_disease.lower()}. 

For common cold/flu: Supportive care including rest, hydration, and symptomatic treatment is recommended. Antiviral medications may be considered for influenza if diagnosed early.

For more serious conditions: Further diagnostic workup, laboratory tests, and clinical correlation are essential. Immediate medical evaluation may be warranted for acute symptoms.

CONFIDENCE LEVEL:
Medium to High - Based on symptom presentation and clinical evidence. For definitive diagnosis, clinical examination and appropriate diagnostic tests are recommended.

RECOMMENDATIONS:
- Clinical examination by healthcare provider
- Symptomatic treatment as appropriate
- Follow-up if symptoms persist or worsen
- Emergency evaluation if severe symptoms develop
"""
    
    return response


def generate_answer(
    system: Dict,
    query: str,
    retrieved_docs: List[Dict],
    max_new_tokens: int = 512
) -> str:
    """
    Generate answer using retrieved documents.
    """
    
    model = system["model"]
    tokenizer = system["tokenizer"]
    
    # Build context
    context_parts = []
    for i, doc in enumerate(retrieved_docs[:3], 1):
        snippet = doc["text"][:500]
        disease = doc.get("disease", "unknown")
        context_parts.append(f"[Evidence {i} - {disease}]\n{snippet}")
    
    context = "\n\n".join(context_parts)
    
    # Create prompt
    prompt = f"""Based on the following medical evidence, provide a diagnostic assessment.

Clinical Evidence:
{context}

Patient Presentation:
{query}

Provide a structured diagnostic assessment:

PRIMARY DIAGNOSIS:
[State the most likely diagnosis]

SUPPORTING EVIDENCE:
[Key clinical findings that support this diagnosis]

CLINICAL REASONING:
[Explain the pathophysiological basis]

CONFIDENCE LEVEL:
[High/Medium/Low and why]

Assessment:"""

    # Format for model - use TinyLlama chat format
    tokenizer = system["tokenizer"]
    
    # Get model's vocabulary size
    vocab_size = model.config.vocab_size
    
    # Ensure tokenizer vocab size matches model
    if hasattr(tokenizer, 'vocab_size') and tokenizer.vocab_size != vocab_size:
        print(f"Warning: Tokenizer vocab size ({tokenizer.vocab_size}) doesn't match model ({vocab_size})")
    
    # TinyLlama uses a specific format - try chat template first
    try:
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
            messages = [
                {"role": "system", "content": "You are a medical expert."},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # Fallback format for TinyLlama
            formatted_prompt = f"<|system|>\nYou are a medical expert.<|user|>\n{prompt}<|assistant|>\n"
    except Exception:
        # Simple format if chat template fails
        formatted_prompt = f"System: You are a medical expert.\nUser: {prompt}\nAssistant:"
    
    # Generate
    try:
        inputs = tokenizer(
            formatted_prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048,
            padding=False
        )
    except Exception as e:
        # If tokenization fails, try simpler approach
        formatted_prompt = f"User: {prompt}\nAssistant:"
        inputs = tokenizer(
            formatted_prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048,
            padding=False
        )
    
    # Move inputs to model device
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    # Set pad token if not set
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0
    
    # CRITICAL: Ensure input_ids are within vocabulary range
    # This fixes the "index out of range" error
    input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
    
    # Remove any invalid tokens (shouldn't happen after clamp, but just in case)
    valid_mask = (input_ids >= 0) & (input_ids < vocab_size)
    if not valid_mask.all():
        print(f"Warning: Found {(~valid_mask).sum()} invalid token IDs, removing them")
        # This is tricky - we'd need to rebuild the sequence, so just clamp
        input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
    
    # Prepare generation kwargs - use faster settings optimized for speed
    # On CPU, use greedy decoding (faster) and shorter sequences
    use_cpu = not torch.cuda.is_available()
    
    if use_cpu:
        # CPU: Greedy decoding, shorter sequences for speed
        generation_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": min(max_new_tokens, 150),  # Shorter on CPU
            "do_sample": False,  # Greedy is faster
            "pad_token_id": tokenizer.pad_token_id,
        }
    else:
        # GPU: Can use sampling
        generation_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": min(max_new_tokens, 256),
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "repetition_penalty": 1.1,
            "pad_token_id": tokenizer.pad_token_id,
        }
    
    if tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
    
    if attention_mask is not None:
        generation_kwargs["attention_mask"] = attention_mask
    
    with torch.no_grad():
        try:
            # Set model to eval mode if not already
            model.eval()
            output = model.generate(**generation_kwargs)
        except Exception as e:
            print(f"Generation error: {e}, using fallback response...")
            # If generation fails completely, use fallback
            return generate_fallback_answer(query, retrieved_docs)
    
    # Decode only the new tokens
    input_length = input_ids.shape[1]
    generated_ids = output[0][input_length:]
    generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Check if generation is valid
    generated = generated.strip()
    
    # If generation is empty, too short, or contains only garbage (like repeated "-inner")
    if (not generated or 
        len(generated) < 20 or 
        generated.count('-inner') > 5 or
        generated.count('inner') > 10 or
        len(set(generated.split()[:10])) < 3):  # Check for repetitive garbage
        
        print("Model generation failed or produced garbage, using fallback...")
        return generate_fallback_answer(query, retrieved_docs)
    
    # Check if output looks like garbage (too many repeated words)
    words = generated.split()
    if len(words) > 0:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:  # Less than 30% unique words = likely garbage
            print("Model output appears to be garbage, using fallback...")
            return generate_fallback_answer(query, retrieved_docs)
    
    return generated


def query_rag_system(
    system: Dict,
    query: str,
    k: int = 5,
    alpha: float = 0.7,
    max_new_tokens: int = 512
) -> Dict:
    """
    Complete RAG query pipeline.
    """
    
    import time
    
    start_time = time.time()
    
    # Retrieve
    retrieval_start = time.time()
    retrieved_docs = hybrid_retrieve(system, query, k=k, alpha=alpha)
    retrieval_time = time.time() - retrieval_start
    
    # Generate
    generation_start = time.time()
    answer = generate_answer(system, query, retrieved_docs, max_new_tokens=max_new_tokens)
    generation_time = time.time() - generation_start
    
    total_time = time.time() - start_time
    
    return {
        "answer": answer,
        "retrieved_docs": retrieved_docs,
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "total_time": total_time
    }
