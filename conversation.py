import json
import time
from typing import Dict, Any, Optional, Generator, List, Tuple

from llm_api import send_to_language_model

# Global store for conversation states
conversation_states = {}

def conversation_stream(conversation_name: str, initial_prompt: str, config: Dict[str, Any], conversation_id: str, api_keys: Optional[Dict[str, str]] = None) -> Generator[str, None, None]:
    """
    Manage and stream conversation flow between multiple agents.
    Yields JSON-encoded strings for Server-Sent Events.
    """
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
        
        # The UI initially shows the opening statement and prompt.
        display_history = f"{opening_statement}\n\nUSER PROMPT: {initial_prompt_str}" if opening_statement else f"USER PROMPT: {initial_prompt_str}"
        yield f"data: {json.dumps({'type': 'history_update', 'history': display_history, 'final_response': ''})}\n\n"
        
        final_agent_response = ""
        agents = conversation_config.get("agents", [])
        
        conversation_history_for_prompt: List[Tuple[str, str]] = []

        for idx, agent in enumerate(agents, 1):
            # NEW: Check for pause state before processing an agent
            while conversation_states.get(conversation_id, {}).get("paused", False):
                # Yield a heartbeat event to keep the connection alive while paused
                yield f"data: {json.dumps({'type': 'heartbeat', 'message': 'paused'})}\n\n"
                time.sleep(1)

            agent_name = agent.get('agent_name', f'Agent_{idx}')
            
            try:
                role = str(agent.get("Role", "")).strip()

                # Build agent identifier for display
                agent_identifier = agent_name
                if role:
                    agent_identifier = f"{agent_name} - {role}"
                
                # Show "Thinking..." message
                thinking_display_block = f"{agent_identifier}\nThinking..."
                thinking_history = display_history + f"\n\n===================\n\n{thinking_display_block}"
                yield f"data: {json.dumps({'type': 'history_update', 'history': thinking_history, 'final_response': ''})}\n\n"

                # Construct the prompt for the LLM
                prompt_parts = []
                prompt_parts.append(f"Your name is {agent_name}")

                hobbie = agent.get("hobbie")
                if hobbie:
                    prompt_parts.append(f"Your hobby is {hobbie}")

                if opening_statement:
                    prompt_parts.append(opening_statement)

                if role:
                    prompt_parts.append(role)

                if not conversation_history_for_prompt:
                    # First agent
                    prompt_parts.append(f"\n{initial_prompt_str}")
                else:
                    # Subsequent agents
                    prompt_parts.append(f"\nThis is the user request: {initial_prompt_str}")
                    

                    if len(conversation_history_for_prompt) == 1:
                        prev_agent_name, prev_response = conversation_history_for_prompt[0]
                        history_block = f"""
{prev_agent_name}: 
{prev_response}
End of {prev_agent_name}'s response
"""
                        prompt_parts.append(history_block)
                    else:
                        for i, (prev_agent_name, prev_response) in enumerate(conversation_history_for_prompt):
                            history_block = f"""
{prev_agent_name}:
{prev_response}
End of {prev_agent_name}'s response
"""
                            prompt_parts.append(history_block)


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

                conversation_history_for_prompt.append((agent_name, response_text))
                final_agent_response = response_text
                
                # Build the display part for this agent
                # The agent_identifier is already created above.
                agent_display_block = "\n".join(part for part in [agent_identifier, response_text] if part)

                # Append this agent's block to the display history
                display_history += f"\n\n===================\n\n{agent_display_block}"
                
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