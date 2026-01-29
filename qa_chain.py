"""QA Chain - Uses vector search + chunk metadata for all queries"""
import os
import re
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from config import OPENAI_API_KEY, TOP_K_RESULTS
from vector_store import ResumeVectorStore

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


class ConversationMemory:
    """Tracks conversation to resolve pronouns like 'his', 'her'"""
    
    def __init__(self):
        self.current_person = None
    
    def resolve(self, query: str, available_names: list[str]) -> tuple[str, Optional[str]]:
        """Resolve pronouns to actual names. Returns (resolved_query, person)"""
        query_lower = query.lower()
        
        # Check if query mentions a specific person
        for name in available_names:
            first_name = name.split()[0].lower()
            if first_name in query_lower:
                self.current_person = name
                return query, name
        
        # Check for pronouns
        pronouns = ['his', 'her', 'their', 'he', 'she', 'him', 'them']
        has_pronoun = any(f" {p} " in f" {query_lower} " or query_lower.startswith(f"{p} ") for p in pronouns)
        
        if has_pronoun and self.current_person:
            first_name = self.current_person.split()[0]
            resolved = query
            for p in ['his', 'her', 'their']:
                resolved = re.sub(rf'\b{p}\b', f"{first_name}'s", resolved, flags=re.IGNORECASE)
            for p in ['he', 'she', 'they', 'him', 'them']:
                resolved = re.sub(rf'\b{p}\b', first_name, resolved, flags=re.IGNORECASE)
            print(f"[PRONOUN] '{query}' -> '{resolved}'")
            return resolved, self.current_person
        
        return query, None
    
    def clear(self):
        self.current_person = None


class ResumeQAChain:
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
        self.memory = ConversationMemory()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM),
            ("human", self.HUMAN)
        ])
    def _needs_full_profile(self, query: str) -> bool:
        """Detect recommendation/analysis queries"""
        patterns = [
            r'\bjob[s]?\b', r'\brole[s]?\b', r'\bposition[s]?\b',
            r'\bfit\b', r'\bsuitable\b', r'\bbest\b', r'\brecommend',
            r'\bcareer\b', r'\bopportunit', r'\bhire\b',
            r'\bqualified\b', r'\bgood for\b', r'\bstrength'
        ]
        return any(re.search(p, q) for p in patterns)

    def _is_aggregation_query(self, query: str) -> bool:
        """Check if query needs ALL people"""
        patterns = [
            r'\b(all|every|everyone)\b',
            r'\blist\b',
            r'\brank\b',
            r'\bcompare\b',
            r'\bhow many\b',
            r'\baverage\b',
            r'\bwho has the (most|highest|lowest)\b',
            r'\bsummarize\b'
        ]
        q = query.lower()
        return any(re.search(p, q) for p in patterns)
    
    def _build_context(self, query: str, person: Optional[str], is_aggregation: bool) -> str:
        """Build context from retrieved chunks"""
        
        if is_aggregation:
            # For aggregation: get more results to cover all people
            docs = self.store.search(query, k=50, person=None)
            
            # Also get metadata for all people to ensure coverage
            all_meta = self.store.get_all_metadata()
            
            # Build context with both chunk content and metadata summaries
            parts = ["=== PEOPLE SUMMARIES ==="]
            for meta in all_meta:
                name = meta.get('person_name', 'Unknown')
                parts.append(f"\n[{name}]")
                parts.append(f"Title: {meta.get('current_title', 'N/A')}")
                parts.append(f"Experience: {meta.get('years_experience', 'N/A')}")
                parts.append(f"Education: {meta.get('education', 'N/A')}")
                parts.append(f"Skills: {meta.get('skills', 'N/A')[:200]}")
                parts.append(f"Companies: {meta.get('companies', 'N/A')}")
                parts.append(f"Work History: {meta.get('work_history', 'N/A')}")
            
            # Add relevant chunks
            if docs:
                parts.append("\n\n=== RELEVANT DETAILS ===")
                seen = set()
                for doc in docs:
                    name = doc.metadata.get('person_name', 'Unknown')
                    content = doc.page_content[:500]
                    key = f"{name}:{content[:100]}"
                    if key not in seen:
                        parts.append(f"\n[{name}]: {content}")
                        seen.add(key)
            
            return "\n".join(parts)
        
        else:
            # Standard query: use vector search with optional person filter
            docs = self.store.search(query, k=TOP_K_RESULTS, person=person)
            
            if not docs:
                return "No relevant information found."
            
            parts = []
            for doc in docs:
                name = doc.metadata.get('person_name', 'Unknown')
                parts.append(f"[{name}]: {doc.page_content}")
            
            return "\n\n".join(parts)
    
    def ask(self, query: str, filter_person: Optional[str] = None) -> dict:
        """Ask a question about the resumes"""
        print(f"\n[QA] Query: {query}")
        
        # Get available people
        people = self.store.get_all_people()
        
        # Resolve pronouns
        resolved_query, detected_person = self.memory.resolve(query, people)
        
        # Use explicit filter or detected person
        effective_person = filter_person or detected_person
        
        # Check if aggregation query
        is_agg = self._is_aggregation_query(resolved_query)
        print(f"[QA] Aggregation: {is_agg}, Person: {effective_person}")
        
        # Build context
        context = self._build_context(resolved_query, effective_person, is_agg)
        print(f"[QA] Context: {len(context)} chars")
        
        # Generate answer
        messages = self.prompt.format_messages(
            current_person=effective_person or "None",
            people=", ".join(people) or "None",
            context=context,
            question=resolved_query
        )
        
        response = self.llm.invoke(messages)
        answer = response.content
        
        # Update memory
        if effective_person:
            self.memory.current_person = effective_person
        
        print(f"[QA] Answer: {answer[:200]}...")
        
        return {
            "answer": answer,
            "resolved_query": resolved_query if resolved_query != query else None,
            "person": effective_person,
            "is_aggregation": is_agg
        }
    
    def get_people(self) -> list[str]:
        return self.store.get_all_people()
    
    def clear_memory(self):
        self.memory.clear()
    
    def get_current_person(self) -> Optional[str]:
        return self.memory.current_person