from typing import TypedDict, List, Optional, Literal


class GraphState(TypedDict):
    """State schema cho LangGraph workflow"""
    
    # Input
    user_message: str
    conversation_context: str
    user_only_context: str
    
    # Classification
    intent: Literal["medical_consultation", "doctor_recommendation", "medicine_inquiry", "general_chat"]
    
    # Symptom checking
    has_symptoms: Optional[bool]
    
    # Context retrieval
    medical_context: Optional[str]
    doctor_context: Optional[str]
    medicine_context: Optional[str]
    
    # Output
    system_prompt: str
    prompt: str
    use_context: bool
