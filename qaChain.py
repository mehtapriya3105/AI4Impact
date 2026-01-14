import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_AI_API_KEY")


def get_resume_qa_chain(pdf_paths: list[str] = ["Resume.pdf"],persist_dir: str = "./chroma_db",):
    documents = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        documents.extend(loader.load())
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200
    )
    
    docs = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    
    vectorstore = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional assistant. Ypu will answer like Priya - the person whose document is uploaded, using first person."),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}")
    ])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True, 
        chain_type_kwargs={"prompt": prompt},
    )
    
    print(qa_chain)
    return qa_chain
