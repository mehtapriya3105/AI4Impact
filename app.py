"""Resume RAG Assistant - Streamlit App"""
import os
import streamlit as st

from vector_store import ResumeVectorStore
from qa_chain import ResumeQAChain
from config import PERSIST_DIRECTORY

st.set_page_config(page_title="Resume RAG", page_icon="📄", layout="wide")


@st.cache_resource
def get_store():
    return ResumeVectorStore(persist_dir=PERSIST_DIRECTORY)


def get_qa():
    if 'qa' not in st.session_state:
        st.session_state.qa = ResumeQAChain(persist_dir=PERSIST_DIRECTORY)
    return st.session_state.qa


def main():
    st.title("📄 Resume RAG Assistant")
    
    store = get_store()
    qa = get_qa()
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Upload Resumes")
        
        files = st.file_uploader("Upload PDF/DOCX", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)
        
        if files:
            for f in files:
                path = f"/tmp/{f.name}"
                with open(path, "wb") as file:
                    file.write(f.getbuffer())
                
                with st.spinner(f"Processing {f.name}..."):
                    success, msg = store.ingest(path)
                
                if success:
                    st.success(msg)
                else:
                    st.warning(msg)
                
                try:
                    os.remove(path)
                except:
                    pass
        
        st.divider()
        
        # Stats
        st.header("📊 Stats")
        people = store.get_all_people()
        
        col1, col2 = st.columns(2)
        col1.metric("People", len(people))
        col2.metric("Chunks", store.count())
        
        if people:
            st.write("**People:**")
            for p in people:
                st.write(f"• {p}")
        
        st.divider()
        
        # Filter
        st.header("🔍 Filter")
        options = ["All"] + people
        selected = st.selectbox("Focus on:", options)
        filter_person = None if selected == "All" else selected
        
        # Current context
        current = qa.get_current_person()
        if current:
            st.info(f"💬 Context: **{current}**")
        
        st.divider()
        
        # Clear buttons
        if st.button("🗑️ Clear All Data"):
            store.clear()
            st.cache_resource.clear()
            if 'qa' in st.session_state:
                del st.session_state.qa
            if 'messages' in st.session_state:
                st.session_state.messages = []
            st.success("Cleared!")
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
                result = qa.ask(prompt, filter_person=filter_person)
                answer = result["answer"]
                
                # Show info
                info = []
                if result.get("person"):
                    info.append(f"Context: {result['person']}")
                if result.get("resolved_query"):
                    info.append(f"Resolved: {result['resolved_query']}")
                if result.get("is_aggregation"):
                    info.append("(aggregation query)")
                if info:
                    st.caption(" | ".join(info))
                
                st.markdown(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
    

if __name__ == "__main__":
    main()