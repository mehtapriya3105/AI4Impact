import streamlit as st
from qaChain import get_resume_qa_chain

st.set_page_config(page_title="Chat")

st.title("Chat")

@st.cache_resource
def load_chain():
    return get_resume_qa_chain(["Letter.pdf" , "Resume.pdf"])

qa_chain = load_chain()


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"query": prompt})
            answer = result["result"]
            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
