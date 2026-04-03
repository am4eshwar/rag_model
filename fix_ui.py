import re

with open('streamlit_app.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Remove emojis and fix wording
code = code.replace('page_icon="??",', '')
code = code.replace('st.title("?? RAG Pipeline")', 'st.title("RAG Intelligence Pipeline")')
code = code.replace('st.title("?? Document Upload & Processing")', 'st.title("Document Upload & Processing")')
code = code.replace('st.title("?? Semantic Query & Comparison")', 'st.title("Semantic Query & Comparison")')
code = code.replace('st.title("??? Similarity Detection")', 'st.title("Similarity Detection")')
code = code.replace('st.button("?? Start Processing Pipeline")', 'st.button("Start Processing Pipeline")')
code = code.replace('st.button("?? Clear Vector Store")', 'st.button("Clear Vector Store")')
code = code.replace('st.button("?? Run Query")', 'st.button("Run Query")')
code = code.replace('st.button("?? Run Global Similarity Analysis")', 'st.button("Run Global Similarity Analysis")')
code = code.replace('st.success("? Groq API Connected")', 'st.success("Groq API Connected successfully")')
code = code.replace('st.error("? Groq API Key Missing")', 'st.error("Groq API Key Missing")')
code = code.replace('label="? Processing Complete!"', 'label="Processing Complete"')

# 2. Change CSS for professional look
old_css = '''    .stButton>button {
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
    }'''

new_css = '''    .stButton>button {
        width: 100%;
        border-radius: 4px;
        height: 3em;
        background-color: #1f2937;
        color: #f9fafb;
        border: 1px solid #374151;
        transition: background-color 0.2s, border-color 0.2s;
        font-family: inherit;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #374151;
        border-color: #3b82f6;
    }'''
code = code.replace(old_css, new_css)

# 3. Replace navigation radio with buttons
old_nav = '''    navigation = st.radio(
        "Navigation",
        ["Upload & Process", "Query & Compare", "Similarity Check"],
        index=0
    )'''

new_nav = '''    if "navigation" not in st.session_state:
        st.session_state.navigation = "Upload & Process"
        
    st.subheader("Features")
    
    if st.button("Document Upload & OCR"):
        st.session_state.navigation = "Upload & Process"
    
    if st.button("Document Semantics & Query"):
        st.session_state.navigation = "Query & Compare"
        
    if st.button("Document Similarity & Plagiarism"):
        st.session_state.navigation = "Similarity Check"
        
    navigation = st.session_state.navigation'''

code = code.replace(old_nav, new_nav)

with open('streamlit_app_new.py', 'w', encoding='utf-8') as f:
    f.write(code)
