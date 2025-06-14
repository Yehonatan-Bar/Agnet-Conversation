from typing import Dict, Any, Optional
import google.generativeai as genai
from openai import OpenAI
import anthropic

from utils import GEMINI_API_KEY_FALLBACK, OPENAI_API_KEY_FALLBACK, ANTHROPIC_API_KEY_FALLBACK

def send_to_openai(prompt: str, api_key: str) -> str:
    """
    Send a prompt to OpenAI model.
    
    Args:
        prompt (str): The input prompt to send to the model
        api_key (str): The API key to use for this request.
        
    Returns:
        str: The model's response text
    """
    if not api_key or "PLACEHOLDER" in api_key:
        return "Error: OpenAI API key is not configured. Please provide it in the UI or set the OPENAI_API_KEY environment variable."
    try:
        # The default client creation should be sufficient.
        # The custom httpx client was causing issues and seems intended
        # for a specific deployment environment (Render).
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="o3-2025-04-16",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"Error with OpenAI API: {str(e)}"

def send_to_claude(prompt: str, model_name: str, api_key: str) -> str:
    """
    Send a prompt to Anthropic Claude model.
    
    Args:
        prompt (str): The input prompt to send to the model.
        model_name (str): The specific Claude model to use.
        api_key (str): The API key to use for this request.
        
    Returns:
        str: The model's response text.
    """
    if not api_key or "PLACEHOLDER" in api_key:
        return "Error: Anthropic API key is not configured. Please provide it in the UI or set the ANTHROPIC_API_KEY environment variable."
    try:
        # The default client creation should be sufficient.
        # The custom httpx client was causing issues and seems intended
        # for a specific deployment environment (Render).
        client = anthropic.Anthropic(api_key=api_key)
        
        # Map friendly names to actual model identifiers
        model_to_use = "claude-opus-4-20250514" # Default
        if model_name:
            name_lower = model_name.lower()
            if "claude-opus-4" in name_lower or "claude" == name_lower:
                 model_to_use = "claude-opus-4-20250514"
            # Add other mappings here if needed in the future

        message = client.messages.create(
            model=model_to_use,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        return message.content[0].text
    except Exception as e:
        return f"Error with Anthropic API: {str(e)}"

def send_to_language_model(
    prompt: str,
    model_name: Optional[str] = None,
    api_keys: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Send a prompt to a language model using either Gemini or OpenAI.
    
    Args:
        prompt (str): The input prompt to send to the model
        model_name (Optional[str]): The specific model to use. If None, uses the default model
        api_keys (Optional[Dict[str, str]]): A dictionary of API keys from the UI.
        
    Returns:
        Dict[str, Any]: Dictionary containing the model response and metadata
    """
    api_keys = api_keys or {}
    # Get key from UI, or fallback to env var / placeholder
    gemini_key = api_keys.get("gemini") or GEMINI_API_KEY_FALLBACK
    openai_key = api_keys.get("openai") or OPENAI_API_KEY_FALLBACK
    anthropic_key = api_keys.get("anthropic") or ANTHROPIC_API_KEY_FALLBACK

    try:
        model_name_lower = model_name.lower() if model_name else ""

        if "o3" in model_name_lower:
            response = send_to_openai(prompt, openai_key)
            return {"response": response}
        elif "claude" in model_name_lower:
            response = send_to_claude(prompt, model_name, anthropic_key)
            return {"response": response}
        else: # Default for Gemini or if "gemini" is in the name
            if not gemini_key or "PLACEHOLDER" in gemini_key:
                return {"error": "Error: Gemini API key is not configured. Please provide it in the UI or set the GEMINI_API_KEY environment variable."}
            
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-2.5-pro-preview-06-05')
            response = model.generate_content(prompt)
            return {"response": response.text}
    except Exception as e:
        return {"error": f"Error in language model communication: {str(e)}"} 