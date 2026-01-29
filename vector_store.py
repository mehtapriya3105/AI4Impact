"""Vector Store - ChromaDB with rich chunk metadata"""
import os
from typing import Optional
import chromadb

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import (
    OPENAI_API_KEY, EMBEDDING_MODEL, PERSIST_DIRECTORY,
    COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS
)
from metadata_extractor import MetadataExtractor, ResumeMetadata

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


class ResumeVectorStore:
    def __init__(self, persist_dir: str = PERSIST_DIRECTORY):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        
        self._embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self._extractor = MetadataExtractor()
        
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " "]
        )
        
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Track content hashes to prevent duplicates
        self._hashes = set()
        self._load_existing_hashes()
        
        print(f"[VectorStore] Loaded {self._collection.count()} chunks")
    
    def _load_existing_hashes(self):
        """Load existing content hashes from collection"""
        try:
            results = self._collection.get(include=["metadatas"])
            if results and results['metadatas']:
                for meta in results['metadatas']:
                    h = meta.get('content_hash', '')
                    if h:
                        self._hashes.add(h)
        except:
            pass
    
    def is_duplicate(self, content_hash: str) -> bool:
        return content_hash in self._hashes
    
    def ingest(self, file_path: str) -> tuple[bool, str]:
        """Ingest a resume file. Returns (success, message)"""
        print(f"\n[INGEST] {file_path}")
        
        # Load document
        try:
            text = self._extractor.load_document(file_path)
        except Exception as e:
            return False, f"❌ Cannot read file: {e}"
        
        if len(text) < 100:
            return False, "❌ File too short"
        
        # Check duplicate
        content_hash = self._extractor.generate_hash(text)
        if self.is_duplicate(content_hash):
            return False, "⚠️ Duplicate document"
        
        # Extract metadata
        metadata = self._extractor.extract(file_path, text)
        if not metadata.person_name:
            return False, "❌ Could not extract person name"
        
        print(f"[INGEST] Person: {metadata.person_name}")
        print(f"[INGEST] Skills: {metadata.skills[:100]}...")
        
        # Create chunks
        docs = [Document(page_content=text)]
        chunks = self._splitter.split_documents(docs)
        print(f"[INGEST] Created {len(chunks)} chunks")
        
        if not chunks:
            return False, "❌ No chunks created"
        
        # Prepare for ChromaDB
        chunk_texts = [c.page_content for c in chunks]
        chunk_ids = [f"{metadata.person_name}_{content_hash[:8]}_{i}" for i in range(len(chunks))]
        
        # Attach FULL metadata to EVERY chunk
        chunk_meta = metadata.to_chunk_metadata()
        chunk_metas = []
        for i in range(len(chunks)):
            meta = chunk_meta.copy()
            meta['chunk_index'] = i
            meta['total_chunks'] = len(chunks)
            chunk_metas.append(meta)
        
        # Embed and store
        embeddings = self._embeddings.embed_documents(chunk_texts)
        
        count_before = self._collection.count()
        self._collection.add(
            ids=chunk_ids,
            documents=chunk_texts,
            embeddings=embeddings,
            metadatas=chunk_metas
        )
        
        # Verify
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        count_after = self._collection.count()
        
        if count_after > count_before:
            self._hashes.add(content_hash)
            return True, f"✅ {metadata.person_name} ({len(chunks)} chunks)"
        
        return False, "❌ Failed to add chunks"
    
    def search(self, query: str, k: int = TOP_K_RESULTS, person: Optional[str] = None) -> list[Document]:
        """Search for relevant chunks"""
        embedding = self._embeddings.embed_query(query)
        
        where = None
        if person:
            where = {"person_name": {"$eq": person}}
        
        try:
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=k,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            print(f"[SEARCH] Filter failed: {e}, retrying without filter")
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
        
        docs = []
        if results['documents'] and results['documents'][0]:
            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                docs.append(Document(page_content=doc, metadata=meta))
        
        return docs
    
    def get_all_people(self) -> list[str]:
        """Get unique person names from chunks"""
        try:
            results = self._collection.get(include=["metadatas"])
            names = set()
            if results and results['metadatas']:
                for meta in results['metadatas']:
                    name = meta.get('person_name', '')
                    if name and name != 'Unknown':
                        names.add(name)
            return sorted(list(names))
        except:
            return []
    
    def get_all_metadata(self) -> list[dict]:
        """Get unique metadata for each person (from first chunk of each)"""
        try:
            results = self._collection.get(include=["metadatas"])
            people = {}
            if results and results['metadatas']:
                for meta in results['metadatas']:
                    name = meta.get('person_name', '')
                    if name and name not in people:
                        people[name] = meta
            return list(people.values())
        except:
            return []
    
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
        self._hashes = set()
        print("[VectorStore] Cleared")