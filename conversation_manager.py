import json
import os
import sys
from typing import Dict, Any, Optional, Generator
import re
import inspect
import importlib.util
import google.generativeai as genai
from openai import OpenAI
import anthropic
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import uuid
import time

# --- API Key Configuration ---
# API keys can be provided via the UI.
# As a fallback, the application will check for environment variables.
# If neither is found, you may edit the placeholder values directly (not recommended).
#
# To set environment variables:
#   - On Windows: `setx GEMINI_API_KEY "your_gemini_key"`
#   - On macOS/Linux: `export GEMINI_API_KEY="your_gemini_key"`

GEMINI_API_KEY_FALLBACK = os.environ.get("GEMINI_API_KEY", "GEMINI_API_KEY_PLACEHOLDER")
OPENAI_API_KEY_FALLBACK = os.environ.get("OPENAI_API_KEY", "OPENAI_API_KEY_PLACEHOLDER")
ANTHROPIC_API_KEY_FALLBACK = os.environ.get("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_PLACEHOLDER")

# Clients are now configured on-demand within the request handling.

app = Flask(__name__)

# Global store for conversation states
conversation_states = {}

class InvalidConversationConfigError(Exception):
    """Custom exception for invalid conversation configuration."""
    pass

def load_conversation_config() -> Optional[Dict]:
    """Load the conversation configuration from JSON file."""
    config_path = os.path.join(os.path.dirname(__file__), 'conversationManager.json')
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None
    except Exception:
        return None

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
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model="o3-2025-04-16",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )
        
        # Extract text from the response output
        for out_item in response.output:
            if hasattr(out_item, "content"):
                for element in out_item.content:
                    if hasattr(element, "text"):
                        return element.text
        return ""
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

def conversation_manager(
    conversation_name: str, 
    initial_prompt: str,
    config: Dict[str, Any]
) -> Dict[str, str]:
    """
    Manage conversation flow between multiple agents.
    
    Args:
        conversation_name (str): Name of the conversation type from config
        initial_prompt (str): Initial user prompt
        config (Dict[str, Any]): The full configuration object.
        
    Returns:
        Dict[str, str]: Contains final model response and conversation history
    """
    try:
        initial_prompt_str = str(initial_prompt) if initial_prompt is not None else ""
        print(f"\n=== Starting New Conversation: {conversation_name} ===")
        print(f"Initial prompt: {initial_prompt_str}\n")
        
        if not config:
            return {
                "final_response": "Error: Configuration is missing.",
                "conversation_history": initial_prompt_str,
                "status": "error"
            }
        
        if conversation_name not in config:
            return {
                "final_response": f"Error: Invalid conversation type: {conversation_name}",
                "conversation_history": initial_prompt_str,
                "status": "error"
            }
        
        conversation_config = config[conversation_name]
        
        # This context is passed to each LLM. It grows with each agent's concise response.
        llm_context = ""
        opening_statement = str(conversation_config.get('opening_statement', '')).strip()
        if opening_statement:
            llm_context = f"{opening_statement}\n\nUSER PROMPT: {initial_prompt_str}"
        else:
            llm_context = f"USER PROMPT: {initial_prompt_str}"
        
        # This is the full history sent to the UI for display.
        display_history = llm_context
        
        final_agent_response = ""
        agents = conversation_config.get("agents", [])
        for idx, agent in enumerate(agents, 1):
            agent_name = agent.get('agent_name', f'Agent_{idx}')
            print(f"\n=== Processing Agent: {agent_name} ===")
            
            try:
                prompt_parts = []
                prefix = str(agent.get("prefix", "")).strip()
                context = str(agent.get("context", "")).strip()

                # Construct the prompt for the LLM
                prompt_parts.append(f"Your name is {agent_name}.\n")
                if prefix:
                    prompt_parts.append(prefix + "\n")
                prompt_parts.append("Here is the conversation so far:\n" + llm_context + "\n")
                if context:
                     prompt_parts.append("Your specific instructions are:\n" + context + "\n")
                prompt_parts.append("Now, provide your response.")

                final_prompt = "\n".join(prompt_parts)
                agent_model = agent.get("model", "").strip() or None
                
                model_response = send_to_language_model(prompt=final_prompt, model_name=agent_model)
                
                response_text = ""
                if isinstance(model_response, dict):
                    if 'error' in model_response: response_text = model_response['error']
                    elif 'response' in model_response: response_text = str(model_response.get('response', ''))
                    else: response_text = str(model_response)
                elif model_response: response_text = str(model_response)
                
                final_agent_response = response_text
                
                # Build the display part for this agent
                agent_identifier = agent_name
                if prefix:
                    agent_identifier = f"{agent_name} - {prefix}"
                
                suffix = str(agent.get("suffix", "")).strip()
                agent_display_block = "\n".join(part for part in [agent_identifier, response_text, suffix] if part)
                
                display_history += f"\n\n===================\n\n{agent_display_block}"
                print("\n=== Updated Conversation History (for display) ===")
                print(f"{display_history}\n")
                
                # CRITICAL CHANGE: Update the LLM context with a *concise summary* of the last turn.
                llm_context += f"\n\nAfter that, {agent_name} responded with: '{response_text}'"

            except Exception as agent_err:
                error_msg = f"Error processing agent {agent_name}: {str(agent_err)}"
                final_agent_response = error_msg
                display_history += f"\n\n===================\n\n{error_msg}"
                print(f"\n=== Error with Agent {agent_name} ===\n{error_msg}")

        # Add closing statement to display history if it exists
        closing_statement = str(conversation_config.get('closing_statement', '')).strip()
        if closing_statement:
            display_history += f"\n\n===================\n\n{closing_statement}"
            
        print("\n=== Conversation Complete ===\n")
        return {
            "final_response": final_agent_response,
            "conversation_history": display_history,
            "status": "success"
        }
        
    except Exception as e:
        error_msg = f"Critical error: {str(e)}"
        print(f"\n=== Critical Error ===\n{error_msg}")
        return {
            "final_response": error_msg,
            "conversation_history": llm_context if 'llm_context' in locals() else "",
            "status": "error"
        }

def conversation_stream(conversation_name: str, initial_prompt: str, config: Dict[str, Any], conversation_id: str, api_keys: Optional[Dict[str, str]] = None) -> Generator[str, None, None]:
    """
    Manage and stream conversation flow between multiple agents.
    Yields JSON-encoded strings for Server-Sent Events.
    """
    # This context is passed to each LLM. It grows with each agent's concise response.
    llm_context = ""
    # This is the full history sent to the UI for display.
    display_history = ""

    try:
        # First, yield the conversation ID so the client can pick it up
        yield f"data: {json.dumps({'type': 'system', 'conversation_id': conversation_id})}\n\n"

        initial_prompt_str = str(initial_prompt) if initial_prompt is not None else ""
        
        if not config:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Error: Configuration is missing.'})}\n\n"
            return
        
        if conversation_name not in config:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Error: Invalid conversation type: {conversation_name}'})}\n\n"
            return
        
        conversation_config = config[conversation_name]
        
        opening_statement = str(conversation_config.get('opening_statement', '')).strip()
        if opening_statement:
            # The LLM sees the opening statement and prompt.
            llm_context = f"{opening_statement}\n\nUSER PROMPT: {initial_prompt_str}"
        else:
            llm_context = f"USER PROMPT: {initial_prompt_str}"
        
        # The UI initially shows the same thing.
        display_history = llm_context
        yield f"data: {json.dumps({'type': 'history_update', 'history': display_history, 'final_response': ''})}\n\n"
        
        final_agent_response = ""
        agents = conversation_config.get("agents", [])
        for idx, agent in enumerate(agents, 1):
            # NEW: Check for pause state before processing an agent
            while conversation_states.get(conversation_id, {}).get("paused", False):
                # Yield a heartbeat event to keep the connection alive while paused
                yield f"data: {json.dumps({'type': 'heartbeat', 'message': 'paused'})}\n\n"
                time.sleep(1)

            agent_name = agent.get('agent_name', f'Agent_{idx}')
            
            try:
                prefix = str(agent.get("prefix", "")).strip()
                context = str(agent.get("context", "")).strip()

                # Build agent identifier for display
                agent_identifier = agent_name
                if prefix:
                    agent_identifier = f"{agent_name} - {prefix}"
                
                # Show "Thinking..." message
                thinking_display_block = f"{agent_identifier}\nThinking..."
                thinking_history = display_history + f"\n\n===================\n\n{thinking_display_block}"
                yield f"data: {json.dumps({'type': 'history_update', 'history': thinking_history, 'final_response': ''})}\n\n"

                # Construct the prompt for the LLM
                prompt_parts = []
                prompt_parts.append(f"Your name is {agent_name}.\n")
                if prefix:
                    prompt_parts.append(prefix + "\n")
                prompt_parts.append("Here is the conversation so far:\n" + llm_context + "\n")
                if context:
                     prompt_parts.append("Your specific instructions are:\n" + context + "\n")
                prompt_parts.append("Now, provide your response.")

                final_prompt = "\n".join(prompt_parts)
                agent_model = agent.get("model", "").strip() or None
                
                model_response = send_to_language_model(prompt=final_prompt, model_name=agent_model, api_keys=api_keys)
                
                response_text = ""
                # Safe extraction of response text
                if isinstance(model_response, dict):
                    if 'error' in model_response:
                        response_text = model_response['error']
                    elif 'response' in model_response:
                        response_text = str(model_response.get('response', ''))
                    else:
                        response_text = str(model_response)
                elif model_response:
                    response_text = str(model_response)

                final_agent_response = response_text
                
                # Build the display part for this agent
                # The agent_identifier is already created above.
                suffix = str(agent.get("suffix", "")).strip()
                agent_display_block = "\n".join(part for part in [agent_identifier, response_text, suffix] if part)

                # Append this agent's block to the display history
                display_history += f"\n\n===================\n\n{agent_display_block}"
                
                # CRITICAL CHANGE: Update the LLM context with a *concise summary* of the last turn.
                llm_context += f"\n\nAfter that, {agent_name} responded with: '{response_text}'"

                yield f"data: {json.dumps({'type': 'history_update', 'history': display_history, 'final_response': ''})}\n\n"
                
            except Exception as agent_err:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Error processing agent {agent_name}: {str(agent_err)}'})}\n\n"
                return

        # Add closing statement to display history if it exists
        closing_statement = str(conversation_config.get('closing_statement', '')).strip()
        if closing_statement:
            display_history += f"\n\n===================\n\n{closing_statement}"
            yield f"data: {json.dumps({'type': 'history_update', 'history': display_history, 'final_response': ''})}\n\n"

        yield f"data: {json.dumps({'type': 'final_response_update', 'final_response': final_agent_response})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': f'Critical error: {str(e)}'})}\n\n"
    finally:
        # Clean up the state when the conversation is over
        if conversation_id in conversation_states:
            print(f"Cleaning up state for conversation {conversation_id}")
            del conversation_states[conversation_id]

@app.route('/stream_conversation', methods=['POST'])
def stream_conversation_route():
    """Endpoint to run a conversation and stream results back."""
    data = request.get_json()
    if not data:
        return Response(f"data: {json.dumps({'type': 'error', 'message': 'Invalid request body.'})}\n\n", mimetype='text/event-stream')

    prompt = data.get('prompt')
    conversation_name = data.get('conversation_name')
    config = data.get('config')
    api_keys = data.get('api_keys', {})

    # Create a unique ID for this conversation and store its state
    conversation_id = str(uuid.uuid4())
    conversation_states[conversation_id] = {"paused": False}
    print(f"Starting new conversation with ID: {conversation_id}")

    return Response(stream_with_context(conversation_stream(conversation_name, prompt, config, conversation_id, api_keys)), mimetype='text/event-stream')

@app.route('/pause_conversation/<conversation_id>', methods=['POST'])
def pause_conversation(conversation_id):
    if conversation_id in conversation_states:
        print(f"Pausing conversation {conversation_id}")
        conversation_states[conversation_id]["paused"] = True
        return jsonify({"status": "success", "message": "Conversation paused."})
    return jsonify({"status": "error", "message": "Conversation not found."}), 404

@app.route('/resume_conversation/<conversation_id>', methods=['POST'])
def resume_conversation(conversation_id):
    if conversation_id in conversation_states:
        print(f"Resuming conversation {conversation_id}")
        conversation_states[conversation_id]["paused"] = False
        return jsonify({"status": "success", "message": "Conversation resumed."})
    return jsonify({"status": "error", "message": "Conversation not found."}), 404

@app.route('/save_config', methods=['POST'])
def save_config():
    """Save the updated configuration to the JSON file."""
    try:
        new_config_data = request.get_json()
        if not isinstance(new_config_data, dict):
            return jsonify({"status": "error", "message": "Invalid data format."}), 400
        
        config_path = os.path.join(os.path.dirname(__file__), 'conversationManager.json')
        
        with open(config_path, 'w') as f:
            json.dump(new_config_data, f, indent=4)
            
        return jsonify({"status": "success", "message": "Configuration saved."})
    except Exception as e:
        print(f"Error saving configuration: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    config_on_load = load_conversation_config() or {}

    # GET request
    conversation_names = list(config_on_load.keys())
    selected_conversation = conversation_names[0] if conversation_names else None
        
    return render_template('index.html', 
                           conversation_names=conversation_names, 
                           selected_conversation=selected_conversation,
                           config=config_on_load, # Pass the dictionary
                           prompt="",
                           result=None)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5001)), debug=True)
