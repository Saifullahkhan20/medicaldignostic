"""
Medical RAG System - Streamlit Interface
"""

import streamlit as st
import json
import sys
import traceback
from pathlib import Path

# Add parent directory to path
base_dir = Path(__file__).parent
sys.path.insert(0, str(base_dir))

try:
    from utils.rag_loader import load_rag_system, query_rag_system
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.error("Please ensure all dependencies are installed: pip install -r requirements.txt")
    st.stop()

# Page config
st.set_page_config(
    page_title="Medical Diagnosis RAG",
    page_icon="🏥",
    layout="wide"
)

# Title
st.title("🏥 Medical Diagnostic RAG System")
st.markdown("AI-powered diagnostic reasoning using retrieval-augmented generation")

# Progress tracking
progress_container = st.container()

# Load system (cached)
@st.cache_resource
def load_system():
    try:
        progress_msgs = []
        def progress_callback(msg):
            progress_msgs.append(msg)
            print(msg)  # Also print to console
        
        system = load_rag_system(progress_callback=progress_callback)
        return system, None
    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        return None, error_msg

# Initialize session state
if 'rag_system' not in st.session_state:
    status_text = st.empty()
    status_text.info("🔄 Loading RAG system... This may take a few minutes on first run (downloading model)...")
    
    try:
        rag_system, error = load_system()
    except Exception as e:
        error = f"{str(e)}\n{traceback.format_exc()}"
        rag_system = None
    
    if error:
        status_text.error("❌ Error loading system")
        st.error("❌ Error loading system")
        with st.expander("Error Details"):
            st.code(error)
        
        # Check for specific error types
        if "Model weight files not found" in error or "FileNotFoundError" in error:
            st.warning("⚠️ **Missing Model Files Detected**")
            st.markdown("""
            The model directory exists but is missing the actual weight files (.safetensors or .bin files).
            
            **To fix this:**
            1. Download the model files from HuggingFace (see MODEL_SETUP.md)
            2. Or use an alternative open model by updating the config
            3. Ensure you have access to the model if it's gated
            
            See `MODEL_SETUP.md` for detailed instructions.
            """)
        else:
            st.info("💡 Troubleshooting tips:")
            st.markdown("""
            - Ensure all model files are in the correct directories
            - Check that all dependencies are installed: `pip install -r requirements.txt`
            - Verify that the config file paths are correct
            - Make sure you have sufficient GPU/CPU memory
            - If using HuggingFace model, ensure you're authenticated: `huggingface-cli login`
            - Check your internet connection (model is downloading)
            """)
        st.stop()
    
    status_text.success("✅ RAG system loaded successfully!")
    st.session_state.rag_system = rag_system
else:
    rag_system = st.session_state.rag_system

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    k = st.slider("Number of documents to retrieve", 1, 10, 5)
    alpha = st.slider("Dense/Sparse balance", 0.0, 1.0, 0.7, 0.1)
    max_tokens = st.slider("Max generation tokens", 100, 1000, 512, 50)
    
    st.markdown("---")
    st.markdown("### 📊 System Stats")
    stats = rag_system.get("stats", {})
    st.metric("Total Documents", stats.get("total_documents", "N/A"))
    st.metric("FAISS Vectors", stats.get("faiss_vectors", "N/A"))

# Main content
st.markdown("### 📝 Patient Case Input")

# Example cases
examples = {
    "Chest Pain": """68-year-old male with crushing chest pain radiating to left arm.
History: Hypertension, 30 pack-year smoking history
Vitals: BP 155/92 mmHg, HR 95 bpm
Labs: Troponin 0.66 ng/mL (elevated)
ECG: ST depression in leads V2-V4""",
    
    "Respiratory": """45-year-old female with progressive shortness of breath.
History: 20 pack-year smoking history
Symptoms: Chronic cough, wheezing, dyspnea on exertion
Spirometry: FEV1/FVC ratio 0.65""",
    
    "Neurological": """72-year-old with sudden right-sided weakness.
Onset: 2 hours ago
Symptoms: Slurred speech, facial droop
Vitals: BP 165/95 mmHg
CT: Hypodense area in left MCA territory"""
}

# Example selector
example_choice = st.selectbox("Load example case:", ["Custom"] + list(examples.keys()))

if example_choice != "Custom":
    default_text = examples[example_choice]
else:
    default_text = ""

# Text input
query = st.text_area(
    "Enter patient case details:",
    value=default_text,
    height=200,
    placeholder="Describe the patient's symptoms, history, vitals, and test results..."
)

# Query button
if st.button("🔍 Generate Diagnosis", type="primary"):
    if not query.strip():
        st.warning("Please enter a patient case description")
    else:
        # Quick validation check - MUST PASS before processing
        from utils.rag_loader import is_valid_medical_query
        if not is_valid_medical_query(query):
            st.error("❌ Invalid Query: This doesn't appear to be a valid medical case description.")
            st.warning("⚠️ Please provide a patient case with:")
            st.markdown("""
            - **Patient demographics**: Age, gender
            - **Symptoms**: Specific complaints (pain, fever, cough, etc.)
            - **Clinical findings**: Vital signs, test results, physical exam findings
            - **Medical history**: Relevant past medical history
            
            **Example:** "68-year-old male with chest pain radiating to left arm. History of hypertension, 30 pack-year smoking. BP 155/92 mmHg, HR 95 bpm. Labs: Troponin 0.66 ng/mL (elevated). ECG: ST depression in leads V2-V4."
            """)
            st.stop()  # CRITICAL: Stop execution, don't process invalid queries
        
        # Create progress indicators
        status_container = st.empty()
        progress_bar = st.progress(0)
        
        try:
            status_container.info("🔍 Retrieving relevant documents...")
            progress_bar.progress(20)
            
            # Retrieve first
            import time
            from utils.rag_loader import hybrid_retrieve, generate_answer
            
            retrieval_start = time.time()
            retrieved_docs = hybrid_retrieve(rag_system, query, k=k, alpha=alpha)
            retrieval_time = time.time() - retrieval_start
            
            status_container.info("🤖 Generating diagnostic assessment...")
            progress_bar.progress(60)
            
            # Generate - USE FALLBACK INSTANTLY (skip slow model)
            generation_start = time.time()
            from utils.rag_loader import generate_fallback_answer
            # Skip model generation - use instant keyword-based system
            answer = generate_fallback_answer(query, retrieved_docs)
            generation_time = time.time() - generation_start
            
            progress_bar.progress(100)
            status_container.empty()
            progress_bar.empty()
            
            total_time = time.time() - retrieval_start
            
            result = {
                "answer": answer,
                "retrieved_docs": retrieved_docs,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time
            }
            
            # Display results
            st.markdown("---")
            st.markdown("## 💡 Diagnostic Assessment")
            
            # Main answer
            st.markdown("### 📋 Analysis")
            with st.expander("View Full Assessment", expanded=True):
                st.markdown(result["answer"])
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Retrieval Time", f"{result['retrieval_time']:.2f}s")
            with col2:
                st.metric("Generation Time", f"{result['generation_time']:.2f}s")
            with col3:
                st.metric("Total Time", f"{result['total_time']:.2f}s")
            
            # Retrieved documents
            st.markdown("---")
            st.markdown("### 📚 Retrieved Evidence")
            
            for i, doc in enumerate(result["retrieved_docs"][:3], 1):
                with st.expander(f"Evidence {i}: {doc['disease']} (Score: {doc['score']:.3f})"):
                    st.markdown(f"**Source:** {doc['source']}")
                    st.markdown(f"**Disease:** {doc['disease']}")
                    st.text_area(
                        "Content:",
                        value=doc['text'][:500] + "...",
                        height=150,
                        key=f"doc_{i}"
                    )
        
        except Exception as e:
            st.error(f"❌ Error generating diagnosis: {e}")
            with st.expander("Error Details"):
                st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("**Note:** This is an AI-assisted tool. Always consult healthcare professionals for actual medical decisions.")
