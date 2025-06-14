import json
import os
import uuid
from flask import request, jsonify, Response, stream_with_context, render_template, Blueprint

from utils import load_conversation_config, save_conversation_config
from conversation import conversation_stream, conversation_states

main_bp = Blueprint('main', __name__)

@main_bp.route('/stream_conversation', methods=['POST'])
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

@main_bp.route('/pause_conversation/<conversation_id>', methods=['POST'])
def pause_conversation(conversation_id):
    if conversation_id in conversation_states:
        print(f"Pausing conversation {conversation_id}")
        conversation_states[conversation_id]["paused"] = True
        return jsonify({"status": "success", "message": "Conversation paused."})
    return jsonify({"status": "error", "message": "Conversation not found."}), 404

@main_bp.route('/resume_conversation/<conversation_id>', methods=['POST'])
def resume_conversation(conversation_id):
    if conversation_id in conversation_states:
        print(f"Resuming conversation {conversation_id}")
        conversation_states[conversation_id]["paused"] = False
        return jsonify({"status": "success", "message": "Conversation resumed."})
    return jsonify({"status": "error", "message": "Conversation not found."}), 404

@main_bp.route('/save_config', methods=['POST'])
def save_config():
    """Save the updated configuration to the JSON file."""
    try:
        new_config_data = request.get_json()
        if not isinstance(new_config_data, dict):
            return jsonify({"status": "error", "message": "Invalid data format."}), 400
        
        save_conversation_config(new_config_data)
            
        return jsonify({"status": "success", "message": "Configuration saved."})
    except Exception as e:
        print(f"Error saving configuration: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@main_bp.route('/', methods=['GET'])
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