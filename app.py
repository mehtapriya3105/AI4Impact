"""Resume RAG - Streamlit App"""
import os
import streamlit as st
from vector_store import ResumeVectorStore
from qa_chain import ResumeQA
from config import PERSIST_DIRECTORY

st.set_page_config(page_title="Resume RAG", page_icon="📄", layout="wide")


@st.cache_resource
def get_store():
    return ResumeVectorStore(persist_dir=PERSIST_DIRECTORY)


def get_qa():
    if 'qa' not in st.session_state:
        st.session_state.qa = ResumeQA(persist_dir=PERSIST_DIRECTORY)
    return st.session_state.qa


def main():
    st.title("📄 Resume RAG Assistant")
    
    store = get_store()
    qa = get_qa()
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Upload Resumes")
        
        files = st.file_uploader("Upload PDF/DOCX", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)
        
        for f in files or []:
            path = f"/tmp/{f.name}"
            with open(path, "wb") as file:
                file.write(f.getbuffer())
            
            with st.spinner(f"Processing {f.name}..."):
                ok, msg = store.ingest(path)
            
            st.success(msg) if ok else st.warning(msg)
            try:
                os.remove(path)
            except:
                pass
        
        st.divider()
        
        # Stats
        people = store.get_people()
        col1, col2 = st.columns(2)
        col1.metric("People", len(people))
        col2.metric("Chunks", store.count())
        
        if people:
            st.write("**People:**")
            for p in people:
                st.write(f"• {p}")
        
        st.divider()
        
        # Filter
        options = ["All"] + people
        selected = st.selectbox("Focus on:", options)
        filter_person = None if selected == "All" else selected
        
        if qa.current_person:
            st.info(f"Context: **{qa.current_person}**")
        
        st.divider()
        
        if st.button("🗑️ Clear All Data"):
            store.clear()
            st.cache_resource.clear()
            st.session_state.clear()
            st.rerun()
        
        if st.button("🔄 Clear Chat"):
            st.session_state.messages = []
            qa.clear_memory()
            st.rerun()
    
    # Chat
    st.header("💬 Ask Questions")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("Ask about resumes..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = qa.ask(prompt, filter_person)
                
                info = []
                if result.get("person"):
                    info.append(f"Context: {result['person']}")
                if result.get("resolved"):
                    info.append(f"→ {result['resolved']}")
                if info:
                    st.caption(" | ".join(info))
                
                st.markdown(result["answer"])
        
        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
    
   

if __name__ == "__main__":
    main()