import streamlit as st
import os
from pathlib import Path
import time
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

# Load RAG modules
from batch_processor import BatchProcessor, ProcessingStatus, BatchStatus
from document_queries import DocumentQuerySystem
from plagiarism_detector import PlagiarismDetector
from config import RAW_DOCS_DIR, DATA_DIR, INDEX_FILE_PATH, CHROMA_PERSIST_DIR

# Load environment variables
load_dotenv()

# ==============================================================================
# UI CONFIGURATION & AESTHETICS
# ==============================================================================
st.set_page_config(
    page_title="RAG Intelligence Pipeline",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #262730;
        color: #ffffff;
        border: 1px solid #4B4B4B;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #4B4B4B;
        border-color: #FF4B4B;
    }
    .card {
        padding: 1.5rem;
        border-radius: 12px;
        background-color: #1e1e26;
        border: 1px solid #2d2d38;
        margin-bottom: 1rem;
    }
    .status-badge {
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .success { background-color: #1a4d2e; color: #a3e635; }
    .processing { background-color: #1e3a8a; color: #93c5fd; }
    .failed { background-color: #7f1d1d; color: #fca5a5; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# INITIALIZATION
# ==============================================================================
@st.cache_resource
def get_orch():
    return {
        "batch": BatchProcessor(),
        "query": DocumentQuerySystem(),
        "similarity": PlagiarismDetector()
    }

orch = get_orch()

# Reset function
def clear_vector_store():
    # Delete FAISS index
    if INDEX_FILE_PATH.exists():
        os.remove(INDEX_FILE_PATH)
    
    # Delete Chroma DB
    if CHROMA_PERSIST_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_PERSIST_DIR)
    
    # Reset objects
    st.cache_resource.clear()
    st.success("Vector Store Cleared! Restarting engine...")
    time.sleep(1)
    st.rerun()

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.title("🧠 RAG Pipeline")
    st.markdown("---")
    
    navigation = st.radio(
        "Navigation",
        ["Upload & Process", "Query & Compare", "Similarity Check"],
        index=0
    )
    
    st.markdown("---")
    st.subheader("Settings")
    
    # API Key Connection Status
    if os.getenv("GROQ_API_KEY"):
        st.success("✅ Groq API Connected")
    else:
        st.error("❌ Groq API Key Missing")
        st.info("Please add `GROQ_API_KEY` to your `.env` file.")
    
    st.markdown("---")
    if st.button("🧹 Clear Vector Store"):
        if st.checkbox("Confirm deletion"):
            clear_vector_store()

# ==============================================================================
# PAGE 1: UPLOAD & PROCESS
# ==============================================================================
if navigation == "Upload & Process":
    st.title("📂 Document Upload & Processing")
    st.write("Upload `.pdf` or `.docx` files here. The pipeline will automatically index them for retrieval.")

    uploaded_files = st.file_uploader(
        "Select files", 
        type=["pdf", "docx"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.info(f"Ready to process {len(uploaded_files)} files.")
        batch_name = st.text_input("Batch Name", value=f"Batch_{datetime.now().strftime('%Y%m%d_%H%M')}")
        
        if st.button("🚀 Start Processing Pipeline"):
            # Save files to RAW_DOCS_DIR
            saved_paths = []
            for uploaded_file in uploaded_files:
                save_path = RAW_DOCS_DIR / uploaded_file.name
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_paths.append(str(save_path))
            
            # Create Batch
            batch_id = orch["batch"].create_batch(
                file_paths=saved_paths,
                batch_name=batch_name
            )
            
            # Processing status
            status_container = st.container()
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.status(f"Processing Batch {batch_id}...", expanded=True) as status:
                def update_ui(batch_job):
                    p = batch_job.progress
                    progress_bar.progress(p)
                    status_text.text(f"Overall Progress: {p*100:.0f}% ({batch_job.completed_documents}/{batch_job.total_documents} completed)")
                
                # Run processing
                orch["batch"].process_batch(
                    batch_id=batch_id, 
                    parallel_workers=2, 
                    progress_callback=update_ui
                )
                status.update(label="✅ Processing Complete!", state="complete")
            
            st.success(f"Batch {batch_id} processed successfully!")

# ==============================================================================
# PAGE 2: QUERY & COMPARE
# ==============================================================================
elif navigation == "Query & Compare":
    st.title("🔍 Semantic Query & Comparison")
    
    # List available documents
    docs = orch["query"].list_all_documents()
    if not docs:
        st.warning("No documents found in vector store. Please upload and process documents first.")
    else:
        doc_names = {d['doc_id']: d['filename'] for d in docs}
        selected_ids = st.multiselect(
            "Select Documents to Query", 
            options=list(doc_names.keys()),
            format_func=lambda x: doc_names[x]
        )
        
        query_text = st.text_area("Enter your query (e.g., 'What are the main conclusions?')")
        
        if st.button("🔍 Run Query"):
            if not selected_ids:
                st.error("Please select at least one document.")
            elif not query_text:
                st.error("Please enter a query.")
            else:
                with st.spinner("Analyzing documents..."):
                    if len(selected_ids) == 1:
                        # Single Query
                        results = orch["query"].query_single_document(selected_ids[0], query_text)
                        st.subheader(f"Results for: {doc_names[selected_ids[0]]}")
                        
                        if not results.results:
                            st.write("No relevant segments found.")
                        else:
                            for i, res in enumerate(results.results):
                                with st.expander(f"Top Result {i+1} (Score: {res.score:.3f})"):
                                    st.write(res.text)
                    else:
                        # Comparison Mode
                        comparison = orch["query"].compare_documents(selected_ids, query_text)
                        st.subheader("Comparison Analysis")
                        
                        cols = st.columns(len(selected_ids))
                        for i, doc_res in enumerate(comparison["documents"]):
                            with cols[i]:
                                st.markdown(f"**{doc_res['filename']}**")
                                if doc_res["results"]:
                                    st.write(doc_res["results"][0]["text"])
                                    st.caption(f"Confidence: {doc_res['results'][0]['score']:.3f}")
                                else:
                                    st.write("*No relevant content found.*")

# ==============================================================================
# PAGE 3: SIMILARITY CHECK
# ==============================================================================
elif navigation == "Similarity Check":
    st.title("🛡️ Similarity Detection")
    st.write("Identify shared content or potential plagiarism across all documents.")
    
    if st.button("📊 Run Global Similarity Analysis"):
        with st.spinner("Comparing all document pairs..."):
            similarity_matrix = orch["similarity"].check_all_submissions(min_similarity=0.4)
            
            if not similarity_matrix:
                st.success("Analysis complete: No significant similarities detected.")
            else:
                st.subheader("High Similarity Flags")
                
                # Format results for a table
                data = []
                for doc_id, similarities in similarity_matrix.items():
                    for sim in similarities:
                        if sim.overall_similarity >= 0.5: # Flag high
                            data.append({
                                "Document A": orch["similarity"]._get_document_info(doc_id).get('filename', doc_id),
                                "Document B": orch["similarity"]._get_document_info(sim.doc_id_2).get('filename', sim.doc_id_2),
                                "Similarity": f"{sim.overall_similarity:.1%}",
                                "Semantic": f"{sim.semantic_similarity:.1%}",
                                "Overlap": f"{sim.ngram_overlap:.1%}"
                            })
                
                if data:
                    df = pd.DataFrame(data).drop_duplicates(subset=["Similarity", "Document A", "Document B"])
                    st.table(df)
                    
                    st.info("Detailed breakdown of suspicious passages can be found in the report logs.")
                else:
                    st.info("Multiple moderate similarities found, but none above high threshold.")
