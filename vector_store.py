"""Vector Store - ChromaDB with metadata in chunks"""
import os
import json
import hashlib
import re
from typing import Optional
import chromadb

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

from config import (
    OPENAI_API_KEY, EMBEDDING_MODEL, LLM_MODEL,
    PERSIST_DIRECTORY, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
)

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


class ResumeVectorStore:
    """Vector store with ALL metadata embedded in each chunk"""
    
    EXTRACT_PROMPT = """Extract resume info as JSON.

IMPORTANT: Find the person's FULL NAME carefully - it may not be on the first line.
Look for patterns like:
- Name after a title (e.g., "TEACHER\\nJohn Smith")
- Name with unusual spacing (e.g., "JohnM. Smith" should be "John M. Smith")
- Name in header/contact section

Return these fields:
- person_name: Full name 
- email, phone, location
- current_title: Current/recent job title
- years_experience: e.g. "3+ years"
- skills: Comma-separated skills
- companies: Comma-separated company names
- education: e.g. "MS CS from Northeastern (GPA: 3.85)"
- work_history: e.g. "Company A (Jan 2024-Present), Company B (2020-2023)"
- summary: 1-2 sentence summary

Document:
{text}

Return ONLY valid JSON:"""

    def __init__(self, persist_dir: str = PERSIST_DIRECTORY):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        
        self._embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self._llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract resume data as JSON only."),
            ("human", self.EXTRACT_PROMPT)
        ])
        self._chain = self._prompt | self._llm
        
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        print(f"[Store] {self._collection.count()} chunks loaded")
    
    def _load_file(self, path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext == '.pdf':
            loader = PyPDFLoader(path)
        elif ext == '.docx':
            loader = Docx2txtLoader(path)
        else:
            loader = TextLoader(path)
        return '\n'.join([d.page_content for d in loader.load()])
    
    def _hash(self, text: str) -> str:
        return hashlib.md5(' '.join(text.lower().split()).encode()).hexdigest()
    
    def _extract_metadata(self, text: str) -> dict:
        try:
            resp = self._chain.invoke({"text": text[:10000]})
            result = resp.content.strip()
            if result.startswith('```'):
                result = '\n'.join(result.split('\n')[1:-1])
            start, end = result.find('{'), result.rfind('}')
            if start != -1 and end != -1:
                result = result[start:end+1]
            return json.loads(result)
        except Exception as e:
            print(f"[Extract] Error: {e}")
            return {}
    
    def _is_duplicate(self, content_hash: str) -> bool:
        try:
            results = self._collection.get(
                where={"content_hash": {"$eq": content_hash}},
                limit=1
            )
            return len(results['ids']) > 0
        except:
            return False
    
    def ingest(self, file_path: str) -> tuple[bool, str]:
        print(f"\n[Ingest] {file_path}")
        
        try:
            text = self._load_file(file_path)
        except Exception as e:
            return False, f"❌ Cannot read: {e}"
        
        if len(text) < 100:
            return False, "❌ Too short"
        
        content_hash = self._hash(text)
        if self._is_duplicate(content_hash):
            return False, "⚠️ Duplicate"
        
        meta = self._extract_metadata(text)
        person_name = meta.get('person_name', '')
        
        # Clean up name if it has weird spacing (e.g., "FarrahM. Bauman" -> "Farrah M. Bauman")
        if person_name:
            # Fix cases like "JohnM." -> "John M."
            person_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', person_name)
            person_name = person_name.strip()
        
        # Fallback: try to find name in first few lines
        if not person_name or person_name == 'Unknown':
            lines = text.split('\n')
            for line in lines[:10]:  # Check first 10 lines
                line = line.strip()
                # Skip empty lines, emails, phones, titles like "TEACHER", "RESUME"
                if not line:
                    continue
                if '@' in line or any(c.isdigit() for c in line[:5]):
                    continue
                if line.upper() == line and len(line.split()) <= 2:  # Skip all-caps titles
                    continue
                if len(line) < 50 and len(line.split()) >= 2:  # Likely a name (2+ words, not too long)
                    # Check it looks like a name (mostly letters)
                    if sum(c.isalpha() or c.isspace() or c == '.' for c in line) / len(line) > 0.8:
                        person_name = line
                        break
        
        if not person_name:
            person_name = "Unknown"
        
        # Get source filename
        source_file = os.path.basename(file_path)
        print(f"[Ingest] Person: {person_name}, File: {source_file}")
        
        docs = self._splitter.split_documents([Document(page_content=text)])
        if not docs:
            return False, "❌ No chunks"
        
        print(f"[Ingest] {len(docs)} chunks")
        
        chunk_meta = {
            "person_name": str(person_name or "Unknown"),
            "source_file": str(source_file),
            "email": str(meta.get('email', '') or ''),
            "phone": str(meta.get('phone', '') or ''),
            "location": str(meta.get('location', '') or ''),
            "current_title": str(meta.get('current_title', '') or ''),
            "years_experience": str(meta.get('years_experience', '') or ''),
            "skills": str(meta.get('skills', '') or ''),
            "companies": str(meta.get('companies', '') or ''),
            "education": str(meta.get('education', '') or ''),
            "work_history": str(meta.get('work_history', '') or ''),
            "summary": str(meta.get('summary', '') or ''),
            "content_hash": content_hash
        }
        
        chunk_ids = [f"{person_name}_{content_hash[:8]}_{i}" for i in range(len(docs))]
        chunk_texts = [d.page_content for d in docs]
        chunk_metas = [{**chunk_meta, "chunk_idx": i} for i in range(len(docs))]
        
        embeddings = self._embeddings.embed_documents(chunk_texts)
        
        self._collection.add(
            ids=chunk_ids,
            documents=chunk_texts,
            embeddings=embeddings,
            metadatas=chunk_metas
        )
        
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        
        return True, f"✅ {person_name} ({len(docs)} chunks)"
    
    def search(self, query: str, k: int = TOP_K, person: str = None) -> list[Document]:
        emb = self._embeddings.embed_query(query)
        where = {"person_name": {"$eq": person}} if person else None
        
        try:
            results = self._collection.query(
                query_embeddings=[emb], n_results=k, where=where,
                include=["documents", "metadatas"]
            )
        except:
            results = self._collection.query(
                query_embeddings=[emb], n_results=k,
                include=["documents", "metadatas"]
            )
        
        docs = []
        if results['documents'] and results['documents'][0]:
            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                docs.append(Document(page_content=doc, metadata=meta))
        return docs
    
    def get_all_chunks(self) -> list[Document]:
        results = self._collection.get(include=["documents", "metadatas"])
        docs = []
        if results['documents']:
            for doc, meta in zip(results['documents'], results['metadatas']):
                docs.append(Document(page_content=doc, metadata=meta))
        return docs
    
    def get_people(self) -> list[str]:
        results = self._collection.get(include=["metadatas"])
        names = set()
        if results['metadatas']:
            for m in results['metadatas']:
                n = m.get('person_name', '')
                if n and n != 'Unknown':
                    names.add(n)
        return sorted(names)
    
    def get_people_with_files(self) -> list[dict]:
        """Get unique people with their source files"""
        results = self._collection.get(include=["metadatas"])
        people = {}
        if results['metadatas']:
            for m in results['metadatas']:
                name = m.get('person_name', '')
                if name and name != 'Unknown' and name not in people:
                    people[name] = {
                        "name": name,
                        "source_file": m.get('source_file', 'Unknown'),
                        "current_title": m.get('current_title', ''),
                    }
        return sorted(people.values(), key=lambda x: x['name'])
    
    def count(self) -> int:
        return self._collection.count()
    
    def clear(self):
        try:
            self._client.delete_collection(COLLECTION_NAME)
        except:
            pass
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        print("[Store] Cleared")