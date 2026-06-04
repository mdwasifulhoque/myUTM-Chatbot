from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, MessageRole
from src.config import LLM_MODEL

# =====================================================================
# ENTERPRISE RAG SYSTEM PROMPT (HIGH-PRECISION DIRECTIVES)
# =====================================================================
SYSTEM_PROMPT = (
    "You are the official myUTM Intelligent Assistant, the premier conversational AI for "
    "Universiti Teknologi Malaysia (UTM). Your sole function is to provide students with precise, "
    "factual information regarding their academic schedules, courses, and campus logistics based "
    "EXCLUSIVELY on the retrieved university data provided in the prompt.\n\n"
    
    "=== 1. CORE BEHAVIORAL CONSTRAINTS (NO META-TALK) ===\n"
    "- You are strictly forbidden from exposing your RAG architecture or internal processes.\n"
    "- NEVER use introductory filler phrases such as 'Based on the provided context', 'According to the data', "
    "'I found the following information', or 'In the given document'.\n"
    "- NEVER use trailing conversational filler such as 'I hope this helps!', 'Let me know if you need more info', "
    "or 'Have a great day!'. Deliver the raw facts directly and instantly stop generating text.\n\n"
    
    "=== 2. ABSOLUTE FACTUAL GROUNDING ===\n"
    "- You must extract answers strictly from the 'University Reference Data'.\n"
    "- Do not use pre-trained external knowledge to guess campus operations, bus times, or library hours.\n"
    "- Cross-reference course codes carefully with their Days, Sections, and Room Locations.\n"
    "- If a location or time is labeled 'TBC', state cleanly that it is 'To Be Confirmed' without speculating.\n\n"
    
    "=== 3. FORMATTING STANDARDS ===\n"
    "- Present scheduling data (Course, Day, Time, Section, Location) using clean, professional bullet points.\n"
    "- Keep the language authoritative, concise, and easy for a student to scan on a mobile interface.\n\n"
    
    "=== 4. EXCEPTION HANDLING (THE FALLBACK PROTOCOL) ===\n"
    "- THE MUTUAL EXCLUSIVITY RULE: If you successfully extract data to answer the user's query, your task is complete. "
    "You must NOT append any apology or 'missing data' text to the bottom of a valid response.\n"
    "- THE MISSING DATA TRIGGER: If the answer to the student's factual query cannot be found ANYWHERE in the "
    "provided reference data, you must halt all data extraction and output ONLY this exact phrase verbatim:\n"
    "'I am sorry, but I do not have that specific information right now. Please check the official myUTM portal "
    "or contact your faculty department.'"
)

# =====================================================================
# SYSTEM PROMPT FOR INSTANT GREETINGS
# =====================================================================
GREETING_PROMPT = (
    "You are the official myUTM Intelligent Assistant for Universiti Teknologi Malaysia (UTM). "
    "Respond to the student's greeting with a warm, welcoming, professional one-to-two sentence opening. "
    "Ask how you can assist them today. "
    "Do not mention any specific classes, dates, times, buses, or missing database errors."
)

class UTMInferenceEngine:
    def __init__(self):
            # Explicitly enforce temperature=0.0 to kill conversational filler and hallucinations
            self.llm = Ollama(
                model=LLM_MODEL, 
                request_timeout=60.0,
                additional_kwargs={"temperature": 0.0}
            )

    def generate_response(self, user_query: str, retrieved_context: str, chat_history: list = None) -> str:
        query_clean = user_query.strip().lower()
        
        # -----------------------------------------------------------------
        # 1. THE INTERCEPTOR: Short-circuit the pipeline for pure greetings
        # -----------------------------------------------------------------
        greetings = ["hello", "hi", "hey", "assalamualaikum", "selamat datang", "selamat pagi"]
        
        if any(query_clean == g or query_clean.startswith(g + " ") for g in greetings) and len(query_clean.split()) <= 3:
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=GREETING_PROMPT),
                ChatMessage(role=MessageRole.USER, content=user_query.strip())
            ]
            try:
                response = self.llm.chat(messages)
                return response.message.content.strip()
            except Exception:
                return "Hello! Welcome to the myUTM Intelligent Assistant. How can I assist you today?"

        # -----------------------------------------------------------------
        # 2. STANDARD RAG PIPELINE: For true structural data validation queries
        # -----------------------------------------------------------------
        context_stripped = retrieved_context.strip() if retrieved_context else "No context provided."
        messages = [ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT)]

        # Map history array properties cleanly into conversational tokens
        if chat_history and len(chat_history) > 1:
            for msg in chat_history[:-1]:
                role = MessageRole.USER if msg["role"] == "user" else MessageRole.ASSISTANT
                messages.append(ChatMessage(role=role, content=msg["content"]))

        # Build final unified layout message payload
        messages.append(ChatMessage(
                    role=MessageRole.USER, 
                    content=(
                        f"University Reference Data:\n{context_stripped}\n\n"
                        f"Student Query: {user_query.strip()}\n\n"
                        f"CRITICAL REMINDER: Output ONLY the raw facts or bullet points requested. "
                        f"Do NOT say 'Based on the context', do NOT use introductory phrases, "
                        f"and do NOT add warnings or trailing disclaimers. Answer immediately:"
                    )
                ))

        try:
            response = self.llm.chat(messages)
            return response.message.content.strip()
        except Exception as e:
            return f"❌ Inference Error: Unable to reach local model engine. Details: {e}"