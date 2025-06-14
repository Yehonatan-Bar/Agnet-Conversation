import os
import random
from typing import Dict, Any, List, Tuple

from flask import Flask
from dotenv import load_dotenv

from llm_api import send_to_language_model
from routes import main_bp
from utils import load_hobbies
from auth import auth_bp, init_auth

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "supersecretkey")

init_auth(app)

app.register_blueprint(auth_bp)
app.register_blueprint(main_bp)


class InvalidConversationConfigError(Exception):
    """Custom exception for invalid conversation configuration."""
    pass


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
        opening_statement = str(conversation_config.get('opening_statement', '')).strip()

        # This is the full history sent to the UI for display.
        display_history = f"{opening_statement}\n\nUSER PROMPT: {initial_prompt_str}" if opening_statement else f"USER PROMPT: {initial_prompt_str}"

        final_agent_response = ""
        agents = conversation_config.get("agents", [])
        
        all_hobbies = load_hobbies()
        if not all_hobbies:
            return {
                "final_response": "Error: Hobbies file could not be loaded or is empty.",
                "conversation_history": initial_prompt_str,
                "status": "error"
            }

        if len(agents) > len(all_hobbies):
            return {
                "final_response": "Error: Not enough unique hobbies for the number of agents.",
                "conversation_history": initial_prompt_str,
                "status": "error"
            }
        
        selected_hobbies = random.sample(all_hobbies, len(agents))
        
        conversation_history_for_prompt: List[Tuple[str, str]] = []

        for idx, agent in enumerate(agents, 1):
            agent_name = agent.get('agent_name', f'Agent_{idx}')
            print(f"\n=== Processing Agent: {agent_name} ===")

            try:
                prompt_parts = []
                
                prompt_parts.append(f"Your name is {agent_name}")
                
                hobbie = selected_hobbies[idx-1]
                prompt_parts.append(f"You are {hobbie}")

                if opening_statement:
                    prompt_parts.append(opening_statement)

                role = str(agent.get("Role", "")).strip()
                if role:
                    prompt_parts.append(f"You are a {role}")

                if not conversation_history_for_prompt:
                    # First agent
                    prompt_parts.append(f"\n{initial_prompt_str}")
                else:
                    # Subsequent agents
                    prompt_parts.append(f"\nThis is the user request: {initial_prompt_str}")
                   
                    for prev_agent_name, prev_response in conversation_history_for_prompt:
                        history_block = f"""
{prev_agent_name}:
{prev_response}
End of {prev_agent_name}'s response
"""
                        prompt_parts.append(history_block)

                    prompt_parts.append(f"\n{agent_name}:")

                final_prompt = "\n".join(prompt_parts)
                agent_model = agent.get("model", "").strip() or None

                model_response = send_to_language_model(prompt=final_prompt, model_name=agent_model)

                response_text = ""
                if isinstance(model_response, dict):
                    if 'error' in model_response:
                        response_text = model_response['error']
                    elif 'response' in model_response:
                        response_text = str(model_response.get('response', ''))
                    else:
                        response_text = str(model_response)
                elif model_response:
                    response_text = str(model_response)
                
                # The model sometimes mimics the "End of..." marker from the prompt.
                # We remove it from the response to avoid showing it in the UI.
                end_marker = f"End of {agent_name}'s response"
                if response_text.strip().endswith(end_marker):
                    end_marker_pos = response_text.rfind(end_marker)
                    response_text = response_text[:end_marker_pos].rstrip()
                
                conversation_history_for_prompt.append((agent_name, response_text))
                final_agent_response = response_text

                # Build the display part for this agent
                agent_identifier = agent_name
                if role:
                    agent_identifier = f"{agent_name} - {role}"

                agent_display_block = "\n".join(part for part in [agent_identifier, response_text] if part)

                display_history += f"\n\n===================\n\n{agent_display_block}"
                print("\n=== Updated Conversation History (for display) ===")
                print(f"{display_history}\n")

            except Exception as agent_err:
                error_msg = f"Error processing agent {agent_name}: {str(agent_err)}"
                final_agent_response = error_msg
                display_history += f"\n\n===================\n\n{error_msg}"
                print(f"\n=== Error with Agent {agent_name} ===\n{error_msg}")

        
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
            "conversation_history": "An error occurred, history not available.",
            "status": "error"
        }


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5001)), debug=True)