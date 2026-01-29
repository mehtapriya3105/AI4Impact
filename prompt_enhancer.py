"""
Prompt Enhancement Module

Enhances user prompts by:
1. Rewriting vague or incomplete prompts
2. Resolving pronouns to actual names based on context
3. Adding context from conversation history
"""

import os
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import LIGHTWEIGHT_LLM_MODEL, OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


class PromptEnhancer:
    """
    Enhances user prompts for better retrieval and response quality
    """
    
    ENHANCEMENT_PROMPT = """
    
    Role:
    You are a prompt enhancement assistant. Your job is to rewrite user queries to be clearer and more specific for a resume/document Q&A system.

    Available People in the System:
    {available_names}

    Conversation History (last few exchanges):
    {conversation_history}

    Current User Query: {user_query}

    Instructions:
    1. If the query uses pronouns (he, she, they, his, her, their) that refer to a person discussed in conversation history, replace them with the actual person's name
    2. If the query is vague or incomplete, make it more specific based on context
    3. If the query asks about "the resume" or "the document" without specifying whose, infer from conversation history
    4. Preserve the original intent of the question
    5. Do NOT add information that wasn't implied in the original query
    6. If the query is already clear and specific, return it mostly unchanged
    7. Format the enhanced query as a clear, complete question
    8. If the user uses any offesive terminology please refrain from answering in a soft manner
    9. If there is any subject related query asked by the user like Web developement - try to identify the relevant skills for that and then try to map based on that skills
    10.If there is any type of output which has a list of dates and other data sort as the most recent first 
    11.If the user asks you to find people with some type of specific skills then rate them with the follwing type : 
        If skill is found in the expirience - highest priority 
        If skill is found in the education - 2nd highest priority 
        If skill is found in the project - medium priority 
        If skill is found anywhere in the resume - least priority 
    13. When asked a question where you think can be answered from the content and other sources give it.
    12. If there is somethign about which there is no information - do not create information on your own 
    

    Examples:
    - Original: "What are his skills?" (after discussing John Doe) → Enhanced: "What are John Doe's skills?"
    - Original : "Who are the people with web developement skills" -> Enhanced: "Who are the people with work in HTML or CSS or Javascrit or MERN or LAMP or PHP or any other web based framework"
    - Original: "Tell me about their education" (after discussing Jane Smith) → Enhanced: "Tell me about Jane Smith's education?"
    - Original: "summarize" (after discussing Bob's resume) → Enhanced: "Can you summarize Bob Johnson's resume?"
    - Original: "What is John's experience?" → Enhanced: "What is John's professional experience?" (already has name, just clarify)
    - Original : "What is the person X worked for?" - "Based on the skills in the resume what real world jobs is this person working?"
    Return ONLY the enhanced query, nothing else. No explanations, no quotes around it."""

    PRONOUN_RESOLUTION_PROMPT = """Based on the conversation history below, identify which person the pronouns in the current query refer to.

Conversation History:
{conversation_history}

Current Query: {user_query}

Available People: {available_names}

If pronouns (he, she, they, his, her, their, him) are used, return the name of the person being referenced.
If no pronouns need resolution or you cannot determine the reference, return "NONE".

Return ONLY the name or "NONE", nothing else."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=LIGHTWEIGHT_LLM_MODEL,
            temperature=0
        )
        
        # FIXED: Renamed to avoid conflict with method name
        self._enhance_prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that enhances search queries. Be concise."),
            ("human", self.ENHANCEMENT_PROMPT)
        ])
        
        self._pronoun_prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You identify pronoun references. Be precise."),
            ("human", self.PRONOUN_RESOLUTION_PROMPT)
        ])
        
        self._enhance_chain = self._enhance_prompt_template | self.llm
        self._pronoun_chain = self._pronoun_prompt_template | self.llm
    
    def resolve_pronouns(
        self,
        query: str,
        conversation_history: list[dict],
        available_names: list[str]
    ) -> Optional[str]:
        """
        Identify which person pronouns in the query refer to
        
        Args:
            query: User's query that may contain pronouns
            conversation_history: List of previous conversation turns
            available_names: List of names in the system
            
        Returns:
            Name of the person referenced, or None
        """
        # Check if query contains pronouns
        pronouns = ['he', 'she', 'they', 'his', 'her', 'their', 'him', 'them']
        query_lower = query.lower()
        
        has_pronouns = any(
            f' {p} ' in f' {query_lower} ' or
            query_lower.startswith(f'{p} ') or
            query_lower.endswith(f' {p}')
            for p in pronouns
        )
        
        if not has_pronouns:
            return None
        
        # Format conversation history
        history_text = self._format_conversation_history(conversation_history)
        
        if not history_text:
            return None
        
        try:
            response = self._pronoun_chain.invoke({
                "conversation_history": history_text,
                "user_query": query,
                "available_names": ', '.join(available_names) if available_names else "No names available"
            })
            
            result = response.content.strip()
            
            if result.upper() == "NONE":
                return None
            
            # Verify the name exists in available names (case-insensitive)
            for name in available_names:
                if result.lower() in name.lower() or name.lower() in result.lower():
                    return name
            
            return result if result else None
            
        except Exception as e:
            print(f"Pronoun resolution error: {e}")
            return None
    
    # FIXED: Renamed method from enhance_prompt to enhance_query
    def enhance_query(
        self,
        query: str,
        conversation_history: list[dict],
        available_names: list[str]
    ) -> str:
        """
        Enhance a user query for better retrieval
        
        Args:
            query: Original user query
            conversation_history: List of previous conversation turns
            available_names: List of names in the system
            
        Returns:
            Enhanced query string
        """
        # Format inputs
        history_text = self._format_conversation_history(conversation_history)
        names_text = ', '.join(available_names) if available_names else "No specific names available"
        
        try:
            response = self._enhance_chain.invoke({
                "available_names": names_text,
                "conversation_history": history_text if history_text else "No previous conversation",
                "user_query": query
            })
            
            enhanced = response.content.strip()
            
            # Remove quotes if the LLM wrapped the response
            if enhanced.startswith('"') and enhanced.endswith('"'):
                enhanced = enhanced[1:-1]
            if enhanced.startswith("'") and enhanced.endswith("'"):
                enhanced = enhanced[1:-1]
            
            return enhanced
            
        except Exception as e:
            print(f"Prompt enhancement error: {e}")
            return query  # Return original on error
    
    def _format_conversation_history(self, history: list[dict]) -> str:
        """Format conversation history for prompt context"""
        if not history:
            return ""
        
        formatted = []
        for turn in history[-5:]:  # Last 5 turns
            role = turn.get('role', 'user')
            content = turn.get('content', '')
            
            # Truncate long messages
            if len(content) > 500:
                content = content[:500] + "..."
            
            formatted.append(f"{role.capitalize()}: {content}")
        
        return '\n'.join(formatted)
    
    def should_enhance(self, query: str) -> bool:
        """
        Determine if a query needs enhancement
        
        Args:
            query: User's query
            
        Returns:
            True if enhancement is recommended
        """
        # Queries that typically need enhancement
        needs_enhancement = [
            len(query.split()) < 4,  # Very short queries
            query.lower().startswith(('what', 'tell', 'show', 'list', 'who', 'find')),
            any(p in query.lower() for p in ['he', 'she', 'they', 'his', 'her', 'their']),
            'resume' in query.lower() and not any(char.isupper() for char in query[query.lower().find('resume')+6:query.lower().find('resume')+20] if query.lower().find('resume')+20 < len(query)),
            query.lower() in ['summarize', 'summary', 'skills', 'experience', 'education'],
            'web' in query.lower() or 'developer' in query.lower() or 'skill' in query.lower()
        ]
        
        return any(needs_enhancement)


class ConversationContextManager:
    """
    Manages conversation history for context-aware responses
    """
    
    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self.history: list[dict] = []
        self.current_person_context: Optional[str] = None
    
    def add_turn(self, role: str, content: str, person_context: Optional[str] = None):
        """Add a conversation turn"""
        self.history.append({
            'role': role,
            'content': content,
            'person_context': person_context
        })
        
        # Update current person context if mentioned
        if person_context:
            self.current_person_context = person_context
        
        # Trim history if needed
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]
    
    def get_history(self, n: Optional[int] = None) -> list[dict]:
        """Get last n conversation turns"""
        n = n or self.max_history
        return self.history[-n * 2:] if self.history else []
    
    def get_context_string(self, n: Optional[int] = None) -> str:
        """Get formatted conversation history string"""
        history = self.get_history(n)
        if not history:
            return ""
        
        formatted = []
        for turn in history:
            role = turn['role'].capitalize()
            content = turn['content']
            if len(content) > 300:
                content = content[:300] + "..."
            formatted.append(f"{role}: {content}")
        
        return '\n'.join(formatted)
    
    def get_current_person(self) -> Optional[str]:
        """Get the person currently being discussed"""
        return self.current_person_context
    
    def clear(self):
        """Clear conversation history"""
        self.history = []
        self.current_person_context = None
    
    def extract_person_from_response(self, response: str, available_names: list[str]) -> Optional[str]:
        """Extract person name mentioned in a response"""
        for name in available_names:
            if name.lower() in response.lower():
                return name
        return None


