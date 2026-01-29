"""Metadata Extractor - Just extracts structured data from resumes"""
import os
import json
import hashlib
from typing import Optional
from dataclasses import dataclass, field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

from config import OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


@dataclass
class ResumeMetadata:
    person_name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    current_title: str = ""
    years_experience: str = ""
    skills: str = ""  # Comma-separated string
    companies: str = ""  # Comma-separated string
    education: str = ""  # Summary string
    work_history: str = ""  # Summary string with dates
    summary: str = ""
    content_hash: str = ""
    
    def to_chunk_metadata(self) -> dict:
        """Convert to flat dict for ChromaDB (no nested objects)"""
        return {
            "person_name": self.person_name or "Unknown",
            "email": self.email or "",
            "phone": self.phone or "",
            "location": self.location or "",
            "current_title": self.current_title or "",
            "years_experience": self.years_experience or "",
            "skills": self.skills or "",
            "companies": self.companies or "",
            "education": self.education or "",
            "work_history": self.work_history or "",
            "summary": self.summary or "",
            "content_hash": self.content_hash or ""
        }


class MetadataExtractor:
    PROMPT = """Extract resume information as JSON.

Return these fields:
- person_name: Full name
- email: Email address
- phone: Phone number  
- location: City, State
- current_title: Current or most recent job title
- years_experience: e.g., "3+ years" or "5 years"
- skills: Comma-separated list of technical skills
- companies: Comma-separated list of companies worked at
- education: Brief summary like "MS Computer Science from Northeastern (GPA: 3.85), BS from MIT"
- work_history: Summary with dates like "Burnes Center (Jan 2024-Present), Google (2020-2023)"
- summary: 1-2 sentence professional summary

Document:
{document_text}

Return ONLY valid JSON:"""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract resume data as JSON. Return only valid JSON."),
            ("human", self.PROMPT)
        ])
        self.chain = self.prompt | self.llm
    
    def load_document(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.docx':
            loader = Docx2txtLoader(file_path)
        else:
            loader = TextLoader(file_path)
        docs = loader.load()
        return '\n'.join([doc.page_content for doc in docs])
    
    def generate_hash(self, content: str) -> str:
        normalized = ' '.join(content.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def extract(self, file_path: str, text: Optional[str] = None) -> ResumeMetadata:
        if text is None:
            text = self.load_document(file_path)
        
        content_hash = self.generate_hash(text)
        
        try:
            response = self.chain.invoke({"document_text": text[:10000]})
            result = response.content.strip()
            
            # Clean JSON
            if result.startswith('```'):
                result = '\n'.join(result.split('\n')[1:-1])
            start = result.find('{')
            end = result.rfind('}')
            if start != -1 and end != -1:
                result = result[start:end+1]
            
            data = json.loads(result)
            
            return ResumeMetadata(
                person_name=data.get('person_name', ''),
                email=data.get('email', ''),
                phone=data.get('phone', ''),
                location=data.get('location', ''),
                current_title=data.get('current_title', ''),
                years_experience=data.get('years_experience', ''),
                skills=data.get('skills', ''),
                companies=data.get('companies', ''),
                education=data.get('education', ''),
                work_history=data.get('work_history', ''),
                summary=data.get('summary', ''),
                content_hash=content_hash
            )
        except Exception as e:
            print(f"Extraction error: {e}")
            # Fallback: try to get name from first line
            first_line = text.split('\n')[0].strip()
            return ResumeMetadata(
                person_name=first_line if len(first_line) < 50 else "Unknown",
                content_hash=content_hash
            )