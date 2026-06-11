import os
import logging
import warnings

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

import streamlit as st
import pdfplumber
import time
from src.inference import UTMInferenceEngine

# =====================================================================
# SHOWCASE PRESENTATION DESIGN (CSS SHIELD)
# =====================================================================
st.set_page_config(
    page_title="myUTM Intelligent Assistant", 
    page_icon="logo/utm_logo.png" 
)

# Robust CSS to override Streamlit's dark mode conflicts and guarantee text visibility
st.markdown("""
    <style>
        /* Force clean light canvas background */
        .stApp {
            background: linear-gradient(180deg, #FFFFFF 0%, #F4F6F9 100%) !important;
        }
        
        /* STRICT TEXT COLOR LOCKS - Prevents the white-on-white text bug */
        .stMarkdown p, .stMarkdown li, .stMarkdown span, .stMarkdown div {
            color: #2D3748 !important;
            font-family: 'Inter', system-ui, sans-serif !important;
        }
        
        /* High-contrast greeting header using UTM Corporate Maroon */
        .premium-header {
            font-weight: 800;
            text-align: center;
            color: #7A1A29 !important;
            font-size: 40px;
            margin-bottom: 0px;
            letter-spacing: -0.5px;
        }
        
        .premium-subheader {
            text-align: center;
            color: #4A5568 !important;
            font-size: 16px;
            margin-top: 6px;
            margin-bottom: 35px;
        }

        /* Glassmorphic interactive question selector cards */
        div.stButton > button {
            background: #FFFFFF !important;
            color: #2D3748 !important;
            border: 1px solid #E2E8F0 !important;
            padding: 22px 18px !important;
            border-radius: 16px !important;
            text-align: left !important;
            min-height: 110px !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.02) !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        
        /* High-fidelity hover mechanics for presentation */
        div.stButton > button:hover {
            border-color: #7A1A29 !important;
            background-color: #FFF5F6 !important;
            color: #7A1A29 !important;
            box-shadow: 0 8px 20px rgba(122, 26, 41, 0.08) !important;
            transform: translateY(-3px);
        }
        
        /* Custom card title font tweaks */
        .card-title {
            font-weight: 700 !important;
            color: #7A1A29 !important;
            margin-bottom: 4px;
        }
        
        /* Polished system chat bubble properties */
        .stChatMessage {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 16px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.01) !important;
            margin-bottom: 15px !important;
        }
        
        /* Rounded floating chat engine input bar */
        .stChatInputContainer {
            border-radius: 24px !important;
            border: 1px solid rgba(122, 26, 41, 0.2) !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04) !important;
            background-color: #FFFFFF !important;
        }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# BACKEND SYSTEM INITIALIZATION & DATA PARSING
# =====================================================================
@st.cache_resource
def load_engine():
    return UTMInferenceEngine()

engine = load_engine()

# Verified Local Resource Path Checks
LOGO_PATH = "logo/utm_logo.png"
use_avatar = LOGO_PATH if os.path.exists(LOGO_PATH) else "assistant"

def extract_context_from_pdf(pdf_filename):
    """
    Advanced layout-aware parser using pdfplumber to cleanly extract text 
    from complex university timetables and tabular course matrices.
    """
    pdf_path = os.path.join("data", pdf_filename)
    
    if not os.path.exists(pdf_path):
        return f"Context missing: The document {pdf_filename} was not found in the database."
        
    try:
        extracted_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text(layout=True)
                if text:
                    extracted_text += text + "\n"
                    
        return extracted_text if extracted_text.strip() else "Error: Document is unreadable or empty."
    except Exception as e:
        return f"Error executing context extraction: {str(e)}"

# Initialize multi-turn chat storage and context tracking states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_context" not in st.session_state:
    st.session_state.active_context = ""

# =====================================================================
# DYNAMIC INTERFACE ROUTING ENGINE (STATE-MACHINE)
# =====================================================================

# PHASE 1: Always render existing historical conversational nodes first
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="👤"): 
            st.markdown(message["content"])
    else:
        bot_avatar = LOGO_PATH if os.path.exists(LOGO_PATH) else "🤖"
        with st.chat_message("assistant", avatar=bot_avatar): 
            st.markdown(message["content"])
            
# PHASE 2: Welcome Screening (Only active when conversation queue is zero)
if not st.session_state.messages:
    st.write("\n")
    if os.path.exists(LOGO_PATH):
        col_l, col_m, col_r = st.columns([1, 0.35, 1])
        with col_m:
            st.image(LOGO_PATH, width='stretch')
            
    st.markdown("<div class='premium-header'>Selamat Datang</div>", unsafe_allow_html=True)
    st.markdown("<div class='premium-subheader'>I am your official myUTM Assistant. How can I guide your lifestyle ecosystem today?</div>", unsafe_allow_html=True)
    
    # 3-Way Grid Presentation Layout
    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        if st.button("🚌 Campus Shuttle\n\nWhen do the campus shuttle buses stop operating?", width='stretch'):
            st.session_state.active_context = extract_context_from_pdf("shuttle_schedule.pdf")
            st.session_state.messages.append({"role": "user", "content": "When do the campus shuttle buses stop operating?"})
            st.rerun()
            
    with col2:
        if st.button("📚 Library Hours\n\nWhat are the operating hours for the PSZ Library?", width='stretch'):
            st.session_state.active_context = extract_context_from_pdf("library_hours.pdf")
            st.session_state.messages.append({"role": "user", "content": "What are the operating hours for the PSZ Library?"})
            st.rerun()
            
    with col3:
        if st.button("📝 Exam Schedules\n\nWhen do the semester final examinations begin?", width='stretch'):
            st.session_state.active_context = extract_context_from_pdf("exam_schedules.pdf")
            st.session_state.messages.append({"role": "user", "content": "When do the semester final examinations begin?"})
            st.rerun()

# PHASE 3: Listen for incoming standard user chat entries
if user_text := st.chat_input("Ask myUTM Assistant..."):
    query_clean = user_text.strip().lower()
    greetings = ["hello", "hi", "hey", "assalamualaikum", "selamat datang", "selamat pagi"]
    
    # ROUTING CHECK: Is the user just saying hello?
    if any(query_clean == g or query_clean.startswith(g + " ") for g in greetings) and len(query_clean.split()) <= 3:
        st.session_state.active_context = "No specific data context requested. The user is just greeting you."
    else:
        # It's an actual question! Load up your evaluation PDFs
        test_docs = ["class_schedule.pdf", "course_list.pdf"]
        all_contexts = []
        
        for doc in test_docs:
            doc_text = extract_context_from_pdf(doc)
            if "Context missing:" not in doc_text and "Error" not in doc_text:
                all_contexts.append(doc_text)
                
        if all_contexts:
            st.session_state.active_context = "\n\n".join(all_contexts)
        else:
            st.session_state.active_context = "No active university schedule or course list files detected."
            
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.rerun()

# PHASE 4: Execution Pipeline (Triggers automatically if the last node is an unreplied user item)
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_query = st.session_state.messages[-1]["content"]
    
    # Render assistant interface component on-the-fly
    with st.chat_message("assistant", avatar=use_avatar):
        with st.spinner("Processing request..."):
            
            # UNIFIED EXECUTION PIPELINE FIX: All tokens route securely through the central engine logic
            ai_response = engine.generate_response(
                user_query=last_query, 
                retrieved_context=st.session_state.active_context,
                chat_history=st.session_state.messages
            )
            
            # PREMIUM UPGRADE: The Typewriter Animation Module
            def response_generator(text_input):
                for word in text_input.split(" "):
                    yield word + " "
                    time.sleep(0.04) # Smooth reading pace tuning
            
            # Stream the string visually onto the interface canvas
            st.write_stream(response_generator(ai_response))
            
    # Commit reply parameters safely into session state and recycle cleanly
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    st.rerun()