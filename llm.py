import ollama

"""
Module: llm.py
Manages the interaction with the Large Language Model (Phi-3).
Focuses on strict prompt engineering to ensure medical accuracy and 
safety by preventing the AI from using outside knowledge.
"""

#Configuration
MODEL_NAME = "phi3"
TIMEOUT_ERROR = "The medical assistant is currently unavailable (Timeout). Please try again later."

def generate_answer(query: str, context: str) -> str:
    """
    Sends clinical context to phi3 and enforces a structured and safe response.

    Handles:
    - Unsupported questions (Non-medical or out of scope questions)
    - No retrieval match (when context does not answer the query)
    - Confidence score generation
    """
    # The system prompt is IMPORTANT since it is the GUARDRAIL for clinical safety.
    system_message = """
    You are a professional Medical AI Assistant. Use ONLY the provided context. 
    Rules:
    1. If the question asks about "dates", "visits", or "history", analyze the context.
    2. If the answer exists in the context, extract it directly.
    3. If the context truly lacks the answer, say "No retrieval match."
    4. Do not use outside knowledge.
    5. Always output:

    Answer: <response>
    Confidence: <High/Medium/Low>
    """
    
    #Building the prompt for phi3: We use chat method
    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"}
    ]
    try:
        #LLM timeout handling
        response =ollama.chat(
            model = MODEL_NAME,
            messages=messages,
            options={
                'temperature': 0.0,            # Zero temperature for deterministic, clinical facts
                'num_predict': 250             # Limit response length to keep it concise
            }
        )
        
        #Extract the content from the LLM response
        answer_text = response['message']['content']
        
        #Ensure the response follows the required output structure
        if "Answer:" not in answer_text:
            return f"Answer: {answer_text}\nConfidence: Low (Format mismatch)"
        
        return answer_text
    except Exception as e:
        #Error Handling: Provides a safe fallback in case of LLM failure or timeout
        return f"Answer: {TIMEOUT_ERROR} \nConfidence: N/A"