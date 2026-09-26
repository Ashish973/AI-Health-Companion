import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage , AIMessage
from database import fetch_recent_messages, log_message
from rag_engine import MentalHealthRAG
from safety_guard import check_crisis_intent

load_dotenv()

SYSTEM_PROMPT_TEMPLATE = """You are a warm, empathetic, and grounded AI Mental Health Companion.
Your core principles:
1. Validate emotions with genuine empathy. Never judge, lecture, or minimize feelings.
2. You are an empathetic conversationalist and reflective journal partner, NOT a licensed doctor or psychiatrist.
3. NEVER make formal clinical diagnoses or prescribe pharmacological treatments.
4. When appropriate, offer gentle evidence-based coping tools (CBT reframing, grounding exercises, mindfulness).
5. Ground your recommendations using the provided Coping Strategies if relevant.

Current Detected User Emotion: {emotion}
Retrieved Self-Care Knowledge:
{rag_context}
"""

class MentalHealthAgent:
    def __init__(self):
        self.rag = MentalHealthRAG()
         api_key = st.secrets["GROQ_API_KEY"]
        model_name = st.secrets["LLM_MODEL"]
        """api_key = os.getenv("GROQ_API_KEY", "").strip()
        model_name = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")"""
        
        if api_key:
            self.llm = ChatGroq(temperature=0.6, groq_api_key=api_key, model_name=model_name)
        else:
            self.llm = None

    def respond(self, user_input: str, detected_emotion: str) -> str:
        # 1. Check crisis protocol
        is_crisis, crisis_msg = check_crisis_intent(user_input)
        if is_crisis:
            log_message("user", user_input, detected_emotion)
            log_message("assistant", crisis_msg, "Crisis Intercepted")
            return crisis_msg

        # 2. Semantic retrieval of coping tactics
        rag_context = self.rag.retrieve_coping_strategies(user_input, k=2)

        # 3. Compile prompt & short-term context
        if not self.llm:
            fallback = f"*(Demo Mode - GROQ_API_KEY not configured)*\n\nI hear you're feeling **{detected_emotion.lower()}**. Thank you for sharing this with me.\n\nHere is a grounding thought from our self-care repository:\n{rag_context}"
            log_message("user", user_input, detected_emotion)
            log_message("assistant", fallback, detected_emotion)
            return fallback

        history = fetch_recent_messages(limit=6)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT_TEMPLATE.format(
                emotion=detected_emotion,
                rag_context=rag_context if rag_context else "None retrieved."
            ))
        ]
        
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["message"]))
            else:
                messages.append(AIMessage(content=msg["message"]))
                
        messages.append(HumanMessage(content=user_input))

        # 4. Invoke LLM
        response = self.llm.invoke(messages).content
        
        # 5. Persist logs
        log_message("user", user_input, detected_emotion)
        log_message("assistant", response, detected_emotion)
        
        return response
