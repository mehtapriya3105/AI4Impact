import streamlit as st
from qaChain import get_resume_qa_chain

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Resume & Letter Chat",
    page_icon="💬",
    layout="wide"
)

# ------------------ Sidebar ------------------
with st.sidebar:
    st.header("📄 About This App")
    st.markdown(
        """
        This chatbot answers questions **only** from the uploaded documents:
        - Resume
        - Admit Letter
        
        It uses:
        - Vector search (Chroma)
        - OpenAI embeddings
        - OpenAI gpt model : 4.0- mini
        """
    )

    st.divider()

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ------------------ Main Title ------------------
st.title("💬 Resume & Letter Chat Assistant")
st.caption("Ask questions about your resume and cover letter. Answers are grounded in the documents.")

# ------------------ Load QA Chain ------------------
@st.cache_resource
def load_chain():
    return get_resume_qa_chain(["Letter.pdf", "Resume.pdf"])

qa_chain = load_chain()

# ------------------ Session State ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------ Chat History ------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------ Chat Input ------------------
if prompt := st.chat_input("Ask anything about your documents..."):
    # User message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching documents..."):
            result = qa_chain.invoke({"query": prompt})
            answer = result["result"]
            sources = result.get("source_documents", [])

            st.markdown(answer)

           

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
