import streamlit as st
from agent import ask

st.set_page_config(
    page_title="Course Assistant AI",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Course Assistant AI")
st.markdown("Ask questions related to your course, AI concepts, or learning topics.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit_user"

user_input = st.text_input("💬 Ask a question:")

if st.button("Ask") and user_input:

    st.session_state.chat_history.append(("You", user_input))

    response = ask(user_input, thread_id=st.session_state.thread_id)

    answer = response.get("answer", "No response")

    st.session_state.chat_history.append(("AI", answer))

st.markdown("### 🧠 Conversation")

for role, msg in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 AI:** {msg}")

if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

st.markdown("---")
st.markdown("Built with ❤️ using Agentic AI (LangGraph + Groq)")
