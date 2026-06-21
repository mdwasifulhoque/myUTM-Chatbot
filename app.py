import os
import logging
import warnings

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

import streamlit as st
import time
from pathlib import Path
from src.inference import UTMInferenceEngine
from src.database import get_vector_index
from src.config import GREETINGS, SIMILARITY_CUTOFF

# =====================================================================
# RESOLVE LOGO PATH FIRST -- needed by set_page_config below, so this
# must be defined before that call, not after it.
# =====================================================================
LOGO_PATH = str(Path(__file__).parent / "logo" / "utm_logo.png")

# =====================================================================
# SHOWCASE PRESENTATION DESIGN (CSS SHIELD)
# =====================================================================
st.set_page_config(
    page_title="myUTM Intelligent Assistant",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🤖"
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
# BACKEND SYSTEM INITIALIZATION & RAG SETUP
# =====================================================================
@st.cache_resource
def load_engine():
    return UTMInferenceEngine()


@st.cache_resource
def load_vector_index():
    """Load the ChromaDB vector index for semantic search."""
    try:
        index = get_vector_index()
        if index is None:
            st.warning("⚠️ Vector database is empty. Please run database build first.")
            return None
        return index
    except Exception as e:
        st.error(f"Error loading vector database: {str(e)}")
        return None


def retrieve_context_from_chromadb(query: str, top_k: int = 3) -> str:
    """
    Query ChromaDB using semantic search to retrieve relevant context.

    Args:
        query: User's question
        top_k: Number of top results to retrieve

    Returns:
        Concatenated context from retrieved documents
    """
    if not st.session_state.vector_index:
        return "Vector database unavailable. Please ensure ChromaDB is initialized."

    try:
        retriever = st.session_state.vector_index.as_retriever(similarity_top_k=top_k)
        results = retriever.retrieve(query)
        print([round(r.score, 3) for r in results])  # TEMP debug line
        # Only keep chunks that are actually relevant -- without this, the
        # retriever returns its top-k closest chunks regardless of whether
        # any of them are a good match for the query.
        relevant_results = [r for r in results if r.score is not None and r.score >= SIMILARITY_CUTOFF]

        if not relevant_results:
            return "No relevant information found in the knowledge base."

        # Concatenate all retrieved document texts
        context = "\n\n".join([node.get_content() for node in relevant_results])
        return context
    except Exception as e:
        return f"Error retrieving context: {str(e)}"


engine = load_engine()

# Verified Local Resource Path Checks
use_avatar = LOGO_PATH if os.path.exists(LOGO_PATH) else "assistant"

# Initialize multi-turn chat storage and context tracking states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_context" not in st.session_state:
    st.session_state.active_context = ""
if "vector_index" not in st.session_state:
    st.session_state.vector_index = load_vector_index()

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
    st.markdown(
        "<div class='premium-subheader'>I am your official myUTM Assistant. How can I guide your lifestyle ecosystem today?</div>",
        unsafe_allow_html=True)

    # 3-Way Grid Presentation Layout - Quick Question Buttons
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        if st.button("🚌 Campus Shuttle\n\nWhen do the campus shuttle buses stop operating?", width='stretch'):
            question = "When do the campus shuttle buses stop operating?"
            st.session_state.active_context = retrieve_context_from_chromadb(question)
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()

    with col2:
        if st.button("📚 Library Hours\n\nWhat are the operating hours for the PSZ Library?", width='stretch'):
            question = "What are the operating hours for the PSZ Library?"
            st.session_state.active_context = retrieve_context_from_chromadb(question)
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()

    with col3:
        if st.button("📝 Exam Schedules\n\nWhen do the semester final examinations begin?", width='stretch'):
            question = "When do the semester final examinations begin?"
            st.session_state.active_context = retrieve_context_from_chromadb(question)
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()

# PHASE 3: Listen for incoming standard user chat entries
if user_text := st.chat_input("Ask myUTM Assistant..."):
    query_clean = user_text.strip().lower()
    greetings = GREETINGS

    # ROUTING CHECK: Is the user just saying hello?
    if any(query_clean == g or query_clean.startswith(g + " ") for g in greetings) and len(query_clean.split()) <= 3:
        st.session_state.active_context = "No specific data context requested. The user is just greeting you."
    else:
        # It's an actual question! Query ChromaDB semantically
        st.session_state.active_context = retrieve_context_from_chromadb(user_text, top_k=3)

    st.session_state.messages.append({"role": "user", "content": user_text})
    st.rerun()

# PHASE 4: Execution Pipeline (Triggers automatically if the last node is an unreplied user item)
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_query = st.session_state.messages[-1]["content"]

    # Render assistant interface component on-the-fly
    with st.chat_message("assistant", avatar=use_avatar):
        with st.spinner("Searching knowledge base..."):

            try:
                # Cap how much history we send -- without this, the prompt
                # grows unbounded over a long conversation.
                MAX_HISTORY_MESSAGES = 6  # roughly the last 3 exchanges
                trimmed_history = st.session_state.messages[-MAX_HISTORY_MESSAGES:]

                # Generate response using RAG pipeline with ChromaDB context
                ai_response = engine.generate_response(
                    user_query=last_query,
                    retrieved_context=st.session_state.active_context,
                    chat_history=trimmed_history
                )


                # PREMIUM UPGRADE: The Typewriter Animation Module
                def response_generator(text_input):
                    for word in text_input.split(" "):
                        yield word + " "
                        time.sleep(0.04)  # Smooth reading pace tuning


                # Stream the string visually onto the interface canvas
                st.write_stream(response_generator(ai_response))

            except Exception as e:
                error_msg = f"❌ Error generating response: {str(e)}"
                st.error(error_msg)
                ai_response = error_msg

    # Commit reply parameters safely into session state and recycle cleanly
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    st.rerun()