"""QA Chain - Vector search + chunk metadata for everything"""
import os
import re
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from config import OPENAI_API_KEY, LLM_MODEL, TOP_K
from vector_store import ResumeVectorStore

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


class ResumeQA:
    SYSTEM = """You are an expert HR assistant and career advisor analyzing resumes.

Your capabilities:
1. **Answer factual questions** - Use ONLY info from CONTEXT
2. **Analyze & recommend** - Based on skills, experience, education in CONTEXT, provide thoughtful analysis
3. **Compare candidates** - Highlight strengths, differences, fit for roles
4. **Suggest job matches** - Based on a person's profile, recommend suitable job titles/roles

Rules:
- Always mention person names
- For factual questions: use ONLY context, say "I don't have that info" if missing
- For recommendations/analysis: reason based on the skills, experience, and education shown in context
- Be specific - cite actual skills, companies, projects from their resume
- For job recommendations, consider: skills match, experience level, industry background, education

Current focus: {focus}
People in system: {people}"""

    HUMAN = """CONTEXT (Resume Data):
{context}

QUESTION: {question}

Provide a helpful, specific answer:"""

    def __init__(self, persist_dir: str = "./chroma_db"):
        self.store = ResumeVectorStore(persist_dir=persist_dir)
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM),
            ("human", self.HUMAN)
        ])
        self.current_person = None
    
    def _needs_all_people(self, query: str) -> bool:
        """Check if query needs data from ALL people"""
        q = query.lower()
        patterns = [
            r'\ball\b', r'\bevery', r'\blist\b', r'\brank\b', 
            r'\bcompare\b', r'\bhow many\b', r'\bwho has\b',
            r'\bsummarize\b', r'\beveryone\b'
        ]
        return any(re.search(p, q) for p in patterns)
    
    def _needs_full_profile(self, query: str) -> bool:
        """Check if query needs full profile analysis (jobs, recommendations, fit)"""
        q = query.lower()
        patterns = [
            r'\bjob[s]?\b', r'\brole[s]?\b', r'\bposition[s]?\b',
            r'\bfit\b', r'\bsuitable\b', r'\bbest\b', r'\brecommend',
            r'\bcareer\b', r'\bopportunit', r'\bhire\b', r'\bcandidate for\b',
            r'\bqualified\b', r'\bgood for\b', r'\bshould apply\b',
            r'\bstrength', r'\bweakness', r'\banalyz'
        ]
        return any(re.search(p, q) for p in patterns)
    
    def _resolve_pronouns(self, query: str, names: list[str]) -> tuple[str, Optional[str]]:
        """Replace pronouns with person name from context"""
        q = query.lower()
        
        # Check for explicit name mention
        for name in names:
            if name.split()[0].lower() in q:
                self.current_person = name
                return query, name
        
        # Check for pronouns
        pronouns = ['his', 'her', 'their', 'he', 'she', 'him', 'them']
        if self.current_person and any(p in q.split() for p in pronouns):
            first = self.current_person.split()[0]
            resolved = query
            for p in ['his', 'her', 'their']:
                resolved = re.sub(rf'\b{p}\b', f"{first}'s", resolved, flags=re.I)
            for p in ['he', 'she', 'him', 'them', 'they']:
                resolved = re.sub(rf'\b{p}\b', first, resolved, flags=re.I)
            return resolved, self.current_person
        
        return query, None
    
    def _build_context(self, query: str, person: Optional[str], all_people: bool) -> str:
        """Build context from chunks"""
        
        # Check if this is a recommendation/analysis query
        needs_profile = self._needs_full_profile(query)
        
        if all_people:
            # Get ALL chunks and dedupe by person for summaries
            chunks = self.store.get_all_chunks()
            
            # Group by person - keep first chunk's metadata as summary
            people_data = {}
            for chunk in chunks:
                name = chunk.metadata.get('person_name', 'Unknown')
                if name not in people_data:
                    people_data[name] = {
                        'meta': chunk.metadata,
                        'chunks': []
                    }
                people_data[name]['chunks'].append(chunk.page_content)
            
            # Build context with metadata summaries + relevant content
            parts = []
            for name, data in people_data.items():
                m = data['meta']
                parts.append(f"\n=== {name} ===")
                parts.append(f"Title: {m.get('current_title', 'N/A')}")
                parts.append(f"Experience: {m.get('years_experience', 'N/A')}")
                parts.append(f"Education: {m.get('education', 'N/A')}")
                parts.append(f"Skills: {m.get('skills', 'N/A')[:300]}")
                parts.append(f"Companies: {m.get('companies', 'N/A')}")
                parts.append(f"Work History: {m.get('work_history', 'N/A')}")
            
            # Also do vector search for relevant details
            docs = self.store.search(query, k=TOP_K)
            if docs:
                parts.append("\n\n=== RELEVANT DETAILS ===")
                seen = set()
                for d in docs:
                    name = d.metadata.get('person_name', 'Unknown')
                    txt = d.page_content[:400]
                    key = f"{name}:{txt[:50]}"
                    if key not in seen:
                        parts.append(f"\n[{name}]: {txt}")
                        seen.add(key)
            
            return "\n".join(parts)
        
        elif needs_profile and person:
            # For job/recommendation queries - get FULL profile for the person
            docs = self.store.search(query, k=30, person=person)
            
            if not docs:
                return "No relevant information found."
            
            # Get metadata from first chunk (all chunks have same metadata)
            m = docs[0].metadata
            
            parts = [f"=== FULL PROFILE: {person} ==="]
            parts.append(f"Current Title: {m.get('current_title', 'N/A')}")
            parts.append(f"Years of Experience: {m.get('years_experience', 'N/A')}")
            parts.append(f"Location: {m.get('location', 'N/A')}")
            parts.append(f"Education: {m.get('education', 'N/A')}")
            parts.append(f"Skills: {m.get('skills', 'N/A')}")
            parts.append(f"Companies Worked At: {m.get('companies', 'N/A')}")
            parts.append(f"Work History: {m.get('work_history', 'N/A')}")
            parts.append(f"Summary: {m.get('summary', 'N/A')}")
            
            # Add detailed chunk content for projects, achievements, etc.
            parts.append(f"\n=== DETAILED EXPERIENCE & PROJECTS ===")
            seen = set()
            for d in docs:
                txt = d.page_content
                if txt[:100] not in seen:
                    parts.append(f"\n{txt}")
                    seen.add(txt[:100])
            
            return "\n".join(parts)
        
        else:
            # Standard vector search
            docs = self.store.search(query, k=TOP_K, person=person)
            
            if not docs:
                return "No relevant information found."
            
            parts = []
            for d in docs:
                name = d.metadata.get('person_name', 'Unknown')
                # Include metadata context with chunk
                meta_ctx = f"[{name} | {d.metadata.get('current_title', '')} | Skills: {d.metadata.get('skills', '')[:100]}]"
                parts.append(f"{meta_ctx}\n{d.page_content}")
            
            return "\n\n".join(parts)
    
    def ask(self, query: str, filter_person: str = None) -> dict:
        """Ask a question"""
        print(f"\n[QA] {query}")
        
        people = self.store.get_people()
        
        # Resolve pronouns
        resolved, detected = self._resolve_pronouns(query, people)
        person = filter_person or detected
        
        # Check query type
        all_people = self._needs_all_people(resolved)
        needs_profile = self._needs_full_profile(resolved)
        print(f"[QA] All people: {all_people}, Profile: {needs_profile}, Person: {person}")
        
        # Build context from chunks
        context = self._build_context(resolved, person, all_people)
        print(f"[QA] Context: {len(context)} chars")
        
        # Ask LLM
        msgs = self.prompt.format_messages(
            focus=person or "None",
            people=", ".join(people) or "None",
            context=context,
            question=resolved
        )
        
        answer = self.llm.invoke(msgs).content
        
        # Update memory
        if person:
            self.current_person = person
        
        return {
            "answer": answer,
            "resolved": resolved if resolved != query else None,
            "person": person
        }
    
    def get_people(self):
        return self.store.get_people()
    
    def clear_memory(self):
        self.current_person = None