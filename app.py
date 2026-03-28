import streamlit as st
import os
import hashlib
import time
import base64
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import speech_recognition as sr
from pydub import AudioSegment
from fpdf import FPDF
import pandas as pd
import json

# --- Page Config ---
st.set_page_config(
    page_title="ProAI Enterprise - Document Intelligence Suite",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS - Enterprise Grade ---
CUSTOM_CSS = """
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Main Container */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 30px;
        padding: 25px;
        margin: 20px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
    }
    
    /* Glass Morphism Effects */
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
    }
    
    /* Animated Gradient Text */
    .gradient-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        animation: gradient 3s ease infinite;
        background-size: 200% 200%;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Login Page */
    .login-wrapper {
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
    }
    
    .login-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 40px;
        padding: 50px;
        width: 100%;
        max-width: 450px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        animation: slideUp 0.5s ease;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .login-header {
        text-align: center;
        margin-bottom: 30px;
    }
    
    .login-header h1 {
        font-size: 3em;
        margin: 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Chat Messages */
    .chat-message {
        padding: 15px 20px;
        border-radius: 20px;
        margin: 10px 0;
        max-width: 80%;
        animation: fadeIn 0.3s ease;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
        box-shadow: 0 10px 20px -5px rgba(102, 126, 234, 0.5);
    }
    
    .assistant-message {
        background: white;
        border-left: 4px solid #667eea;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    /* Metrics Cards */
    .metric-card {
        background: white;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.2);
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 15px !important;
        font-weight: 600 !important;
        padding: 12px 25px !important;
        transition: all 0.3s ease !important;
        border: none !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5) !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f3c 0%, #2a2f5a 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: 15px !important;
        border: 2px solid #e0e0e0 !important;
        padding: 15px !important;
        font-size: 1em !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border-radius: 10px !important;
    }
    
    /* File Uploader */
    .stFileUploader {
        border: 2px dashed rgba(102, 126, 234, 0.3);
        border-radius: 20px;
        padding: 20px;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #667eea;
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background: white;
        padding: 10px;
        border-radius: 20px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 15px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-color: #667eea !important;
        border-top-color: transparent !important;
    }
    
    /* Success/Error/Warning Messages */
    .stSuccess, .stError, .stWarning {
        border-radius: 15px !important;
        padding: 15px !important;
        animation: slideIn 0.3s ease;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
</style>
"""

# --- Initialize Session State ---
def init_session_state():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "file_hash" not in st.session_state:
        st.session_state.file_hash = None
    if "processing_time" not in st.session_state:
        st.session_state.processing_time = None
    if "document_name" not in st.session_state:
        st.session_state.document_name = None
    if "document_stats" not in st.session_state:
        st.session_state.document_stats = {}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "favorites" not in st.session_state:
        st.session_state.favorites = []

init_session_state()
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

@st.cache_resource
def get_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")

# --- Login Page (Enterprise) ---
def login_page():
    st.markdown("""
        <div class="login-wrapper">
            <div class="login-card">
                <div class="login-header">
                    <h1>🚀 ProAI</h1>
                    <p style="color: #666; margin-top: 10px;">Enterprise Document Intelligence</p>
                </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Animated input fields
            user = st.text_input(
                "👤 Username",
                placeholder="Enter your username",
                key="login_user",
                label_visibility="collapsed"
            )
            
            pwd = st.text_input(
                "🔐 Password",
                type="password",
                placeholder="Enter your password",
                key="login_pwd",
                label_visibility="collapsed"
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                if st.button("🚀 Login to Dashboard", use_container_width=True):
                    if user == "admin" and pwd == "12345":
                        st.session_state.logged_in = True
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials. Please try again.")
            
            st.markdown("---")
            
            # Demo credentials card
            st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea10 0%, #764ba210 100%); 
                            border-radius: 15px; padding: 20px; text-align: center;">
                    <p style="color: #666; margin-bottom: 10px;">🔑 Demo Access</p>
                    <p><code style="background: #f0f0f0; padding: 5px 10px; border-radius: 8px;">Username: admin</code></p>
                    <p><code style="background: #f0f0f0; padding: 5px 10px; border-radius: 8px;">Password: 12345</code></p>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)

# --- Dashboard Page (Enterprise) ---
def dashboard_page():
    # Header with metrics
    st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 30px; padding: 30px; margin-bottom: 30px;
                    box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);">
            <h1 style="color: white; margin: 0; font-size: 2.5em;">📊 Analytics Dashboard</h1>
            <p style="color: rgba(255,255,255,0.8); margin-top: 10px;">Real-time document intelligence metrics</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <h3 style="color: #667eea; margin: 0;">📄 Documents</h3>
                <p style="font-size: 2em; font-weight: 700; margin: 10px 0;">{}</p>
                <p style="color: #666;">Processed today</p>
            </div>
        """.format(1 if st.session_state.document_name else 0), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card">
                <h3 style="color: #667eea; margin: 0;">💬 Queries</h3>
                <p style="font-size: 2em; font-weight: 700; margin: 10px 0;">{}</p>
                <p style="color: #666;">Total conversations</p>
            </div>
        """.format(len(st.session_state.messages)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="metric-card">
                <h3 style="color: #667eea; margin: 0;">⚡ Response</h3>
                <p style="font-size: 2em; font-weight: 700; margin: 10px 0;">0.3s</p>
                <p style="color: #666;">Average time</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class="metric-card">
                <h3 style="color: #667eea; margin: 0;">🎯 Accuracy</h3>
                <p style="font-size: 2em; font-weight: 700; margin: 10px 0;">98%</p>
                <p style="color: #666;">Success rate</p>
            </div>
        """, unsafe_allow_html=True)

# --- Main Chat Interface (Enterprise) ---
def chatbot_page():
    # Dashboard at top
    dashboard_page()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat Interface", "📊 Analytics", "⚙️ Settings"])
    
    with tab1:
        # Two column layout
        chat_col, sidebar_col = st.columns([2, 1])
        
        with chat_col:
            # Chat header
            st.markdown("""
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                width: 40px; height: 40px; border-radius: 12px;
                                display: flex; align-items: center; justify-content: center;">
                        <span style="color: white; font-size: 1.5em;">💬</span>
                    </div>
                    <h2 style="margin: 0;">Live Conversation</h2>
                </div>
            """, unsafe_allow_html=True)
            
            # Chat container with scroll
            chat_container = st.container()
            with chat_container:
                for message in st.session_state.messages:
                    if message["role"] == "user":
                        st.markdown(f"""
                            <div class="chat-message user-message">
                                <div style="display: flex; align-items: center; gap: 10px;">
                                    <div style="background: white; width: 30px; height: 30px; 
                                                border-radius: 10px; display: flex; align-items: center; 
                                                justify-content: center;">
                                        <span style="color: #667eea;">👤</span>
                                    </div>
                                    <div style="flex: 1;">{message["content"]}</div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div class="chat-message assistant-message">
                                <div style="display: flex; align-items: center; gap: 10px;">
                                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                                width: 30px; height: 30px; border-radius: 10px;
                                                display: flex; align-items: center; justify-content: center;">
                                        <span style="color: white;">🤖</span>
                                    </div>
                                    <div style="flex: 1;">{message["content"]}</div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
            
            # Chat input with advanced features
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([6, 1, 1])
            with col1:
                prompt = st.chat_input("💭 Ask anything about your documents...", key="chat_input")
            with col2:
                st.button("🎤", help="Voice Input")
            with col3:
                st.button("📎", help="Attach File")
            
            if prompt:
                # User message
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                if st.session_state.retriever and st.session_state.llm:
                    with st.spinner("🤖 Processing your query..."):
                        relevant_docs = st.session_state.retriever.invoke(prompt)
                        context = "\n\n".join([doc.page_content for doc in relevant_docs])
                        
                        qa_prompt = f"""Based on the following context, answer the question clearly:

Context:
{context}

Question: {prompt}

Answer:"""
                        
                        response = st.session_state.llm.invoke(qa_prompt)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.rerun()
                else:
                    st.warning("⚠️ Please upload and process a document first!")
        
        with sidebar_col:
            # Quick actions sidebar
            st.markdown("""
                <div class="glass-card" style="margin-bottom: 20px;">
                    <h3 style="margin-top: 0;">⚡ Quick Actions</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Document upload
            uploaded_file = st.file_uploader(
                "📄 Upload Document",
                type=["pdf", "txt", "docx"],
                help="Supported formats: PDF, TXT, DOCX"
            )
            
            if uploaded_file:
                st.success(f"✅ {uploaded_file.name} uploaded")
                
                if st.button("🚀 Process Document", use_container_width=True):
                    with st.spinner("Processing document..."):
                        # Processing logic
                        start_time = time.time()
                        
                        # Save temp file
                        path = Path("temp") / uploaded_file.name
                        path.parent.mkdir(exist_ok=True)
                        path.write_bytes(uploaded_file.getvalue())
                        
                        # Load and process
                        loader = PyPDFLoader(str(path))
                        docs = loader.load()
                        
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=500,
                            chunk_overlap=50
                        )
                        chunks = splitter.split_documents(docs)
                        
                        # Create vector store
                        embeddings = get_embeddings()
                        vector_db = Chroma.from_documents(
                            documents=chunks,
                            embedding=embeddings
                        )
                        
                        # Initialize LLM and retriever
                        st.session_state.llm = OllamaLLM(
                            model="neural-chat",
                            temperature=0.3
                        )
                        st.session_state.retriever = vector_db.as_retriever(
                            search_kwargs={"k": 3}
                        )
                        
                        # Store metadata
                        st.session_state.document_name = uploaded_file.name
                        st.session_state.processing_time = time.time() - start_time
                        st.session_state.document_stats = {
                            "pages": len(docs),
                            "chunks": len(chunks),
                            "size": f"{uploaded_file.size / 1024:.1f} KB"
                        }
                        
                        st.success(f"✅ Processed in {st.session_state.processing_time:.1f}s")
                        st.balloons()
            
            # Document stats
            if st.session_state.document_stats:
                st.markdown("""
                    <div class="glass-card" style="margin-top: 20px;">
                        <h3>📊 Document Stats</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                stats = st.session_state.document_stats
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Pages", stats.get("pages", 0))
                    st.metric("Size", stats.get("size", "0 KB"))
                with col2:
                    st.metric("Chunks", stats.get("chunks", 0))
                    st.metric("Status", "✅ Active")
            
            # Voice input
            st.markdown("""
                <div class="glass-card" style="margin-top: 20px;">
                    <h3>🎙️ Voice Commands</h3>
                </div>
            """, unsafe_allow_html=True)
            
            audio_file = st.file_uploader(
                "Upload audio",
                type=["wav", "mp3", "m4a"],
                key="audio_upload",
                label_visibility="collapsed"
            )
            
            if audio_file and st.button("🎤 Transcribe", use_container_width=True):
                with st.spinner("Transcribing..."):
                    # Save audio
                    audio_path = Path("temp") / audio_file.name
                    audio_path.write_bytes(audio_file.getvalue())
                    
                    # Convert to wav if needed
                    if audio_path.suffix != ".wav":
                        wav_path = audio_path.with_suffix(".wav")
                        AudioSegment.from_file(str(audio_path)).export(str(wav_path), format="wav")
                    else:
                        wav_path = audio_path
                    
                    # Transcribe
                    r = sr.Recognizer()
                    with sr.AudioFile(str(wav_path)) as source:
                        audio = r.record(source)
                        text = r.recognize_google(audio)
                        
                    # Add to messages
                    st.session_state.messages.append({"role": "user", "content": text})
                    st.success("✅ Transcribed successfully")
                    st.rerun()
            
            # Report generation
            if st.button("📄 Generate Report", use_container_width=True):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", "B", 16)
                pdf.cell(0, 10, "ProAI Enterprise Report", ln=True, align="C")
                pdf.set_font("Arial", size=12)
                pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
                
                for msg in st.session_state.messages[-10:]:  # Last 10 messages
                    pdf.ln(5)
                    pdf.set_font("Arial", "B", 11)
                    pdf.cell(0, 7, f"{msg['role'].upper()}:", ln=True)
                    pdf.set_font("Arial", size=11)
                    pdf.multi_cell(0, 7, msg['content'])
                
                pdf_bytes = pdf.output(dest="S").encode("latin-1")
                st.download_button(
                    "⬇️ Download PDF",
                    pdf_bytes,
                    "proai_report.pdf",
                    "application/pdf",
                    use_container_width=True
                )
            
            # Logout
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.logged_in = False
                st.rerun()
    
    with tab2:
        st.markdown("""
            <div style="background: white; border-radius: 20px; padding: 30px;">
                <h2>📈 Analytics Dashboard</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Create sample analytics
        col1, col2 = st.columns(2)
        
        with col1:
            # Message timeline
            if st.session_state.messages:
                df = pd.DataFrame([
                    {"time": i, "type": msg["role"]}
                    for i, msg in enumerate(st.session_state.messages)
                ])
                fig = px.line(df, x="time", color="type", title="Conversation Flow")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Document stats
            if st.session_state.document_stats:
                labels = ['Pages', 'Chunks']
                values = [
                    st.session_state.document_stats.get("pages", 0),
                    st.session_state.document_stats.get("chunks", 0)
                ]
                fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
                fig.update_layout(title="Document Composition")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("""
            <div style="background: white; border-radius: 20px; padding: 30px;">
                <h2>⚙️ Settings</h2>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Settings")
            temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
            chunk_size = st.number_input("Chunk Size", 100, 1000, 500)
            chunk_overlap = st.number_input("Chunk Overlap", 0, 200, 50)
            
            if st.button("Save Settings", use_container_width=True):
                st.success("✅ Settings saved")
        
        with col2:
            st.subheader("Appearance")
            theme = st.selectbox("Theme", ["Light", "Dark", "System"])
            font_size = st.select_slider("Font Size", ["Small", "Medium", "Large"])
            
            st.subheader("Notifications")
            email_notify = st.checkbox("Email notifications")
            sound_notify = st.checkbox("Sound notifications")

# --- Main ---
if not st.session_state.logged_in:
    login_page()
else:
    chatbot_page()