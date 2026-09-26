import streamlit as st
import pandas as pd
import plotly.express as px
from database import init_db, log_journal_entry, fetch_mood_analytics
from emotion_analyzer import analyze_emotion
from llm_agent import MentalHealthAgent

st.set_page_config(
    page_title="AI Mental Health Companion",
    page_icon="🧠",
    layout="wide"
)

# Initialize database
init_db()

@st.cache_resource
def get_agent():
    return MentalHealthAgent()

agent = get_agent()

# Sidebar Navigation
st.sidebar.title("🌿 Mindful Companion")
menu = st.sidebar.radio("Navigation", ["💬 Reflective Chat", "📓 Guided Journal", "📊 Emotional Analytics", "🆘 Crisis Resources"])

st.sidebar.markdown("---")
st.sidebar.caption("🔒 **Privacy Notice**: All logs and embeddings run locally on your system.")

# ----------------- PAGE 1: CHAT -----------------
if menu == "💬 Reflective Chat":
    st.title("Reflective Space")
    st.write("Share whatever is on your mind. This space is private, non-judgmental, and here to support you.")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Hello. I am here to listen. How are you feeling today?"}
        ]

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Type your message here..."):
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Emotion extraction
        analysis = analyze_emotion(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Reflecting..."):
                reply = agent.respond(user_input, analysis["primary_emotion"])
                st.markdown(reply)
                
        st.session_state.chat_messages.append({"role": "assistant", "content": reply})

# ----------------- PAGE 2: JOURNAL -----------------
elif menu == "📓 Guided Journal":
    st.title("Daily Thought & Mood Journal")
    st.write("Externalizing thoughts reduces mental friction. Log your daily thoughts below.")

    with st.form("journal_form", clear_on_submit=True):
        journal_text = st.text_area("Write your entry here...", height=180)
        self_stress = st.slider("Self-Assessed Stress Level (1 = Completely Calm, 10 = Severe Distress)", 1, 10, 5)
        submitted = st.form_submit_button("Save Journal Entry")

        if submitted and journal_text.strip():
            metrics = analyze_emotion(journal_text)
            log_journal_entry(
                entry_text=journal_text,
                emotion=metrics["primary_emotion"],
                score=metrics["sentiment_score"],
                stress=self_stress
            )
            st.success(f"Entry saved! Detected Context: **{metrics['primary_emotion']}** (Sentiment Score: {metrics['sentiment_score']:.2f})")

# ----------------- PAGE 3: ANALYTICS -----------------
elif menu == "📊 Emotional Analytics":
    st.title("Emotional Trends & Insights")
    data = fetch_mood_analytics()
    
    if not data:
        st.info("No journal entries recorded yet. Write your first entry in the 'Guided Journal' tab to view trends.")
    else:
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Journal Logs", len(df))
        with col2:
            st.metric("Avg Stress Rating", f"{df['stress_level'].mean():.1f} / 10")
        with col3:
            dominant_emotion = df['primary_emotion'].mode()[0]
            st.metric("Predominant State", dominant_emotion)

        st.markdown("---")
        
        # Stress over time
        fig_stress = px.line(
            df, x="timestamp", y="stress_level",
            title="Stress Trajectory Over Time",
            markers=True, line_shape="spline",
            labels={"stress_level": "Stress Index (1-10)", "timestamp": "Date"}
        )
        st.plotly_chart(fig_stress, use_container_width=True)

        # Emotion distribution
        fig_pie = px.pie(
            df, names="primary_emotion",
            title="Emotional State Distribution",
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# ----------------- PAGE 4: CRISIS RESOURCES -----------------
elif menu == "🆘 Crisis Resources":
    st.title("Emergency & Professional Support")
    st.error("If you or someone you know is in immediate danger or severe psychological distress, please contact dedicated emergency hotlines.")
    
    st.markdown("""
    ### 24/7 Crisis Helplines
    - **Tele-MANAS (Govt of India):** `14416` or `1800-891-4416` (Toll-free, multilingual)
    - **KIRAN Mental Health Line:** `1800-599-0019`
    - **Vandrevala Foundation Helpline:** `+91 9999 666 555`
    - **National Suicide Prevention Lifeline (USA):** `988`
    - **Crisis Text Line:** Text `HOME` to `741741`
    
    ### Immediate Grounding Anchor
    When experiencing acute panic or emotional flooding:
    1. **Breathe:** Use the 4-4-4 Box Breathing method (Inhale 4s, Hold 4s, Exhale 4s).
    2. **Ground:** Place both bare feet on the floor and focus entirely on the physical contact with the ground.
    3. **Disconnect:** Step away from all digital screens for at least 15 minutes.
    """)