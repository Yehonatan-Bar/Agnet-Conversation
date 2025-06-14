import json
import os
from typing import Dict, Optional
from pathlib import Path
from flask_login import current_user
from dotenv import load_dotenv

# --- API Key Management ---

# Fallback to environment variables if a UI-provided key isn't available.
# It's recommended to set these in your `.env` file for local development.
GEMINI_API_KEY_FALLBACK = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_PLACEHOLDER")
OPENAI_API_KEY_FALLBACK = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_PLACEHOLDER")
ANTHROPIC_API_KEY_FALLBACK = os.getenv("ANTHROPIC_API_KEY", "YOUR_ANTHROPIC_API_KEY_PLACEHOLDER")

# --- Constants ---
CONVERSATIONS_DIR = "conversations"

BASE_DIR = Path(CONVERSATIONS_DIR)

def _get_user_specific_config_path() -> Path:
    """Gets the path to the conversation config file for the current user."""
    user_id = "anonymous"
    if current_user and current_user.is_authenticated:
        user_id = current_user.get_id()
    
    path = BASE_DIR / user_id
    path.mkdir(parents=True, exist_ok=True)
    return path / 'conversationManager.json'

def load_conversation_config() -> Optional[Dict]:
    """Load the conversation configuration from the user-specific JSON file."""
    config_path = _get_user_specific_config_path()
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        # Create a default config file if it doesn't exist for the user
        default_config = {}
        save_conversation_config(default_config)
        return default_config
    except json.JSONDecodeError:
        return None
    except Exception:
        return None

def save_conversation_config(data: Dict):
    """Saves the conversation configuration to the user-specific JSON file."""
    config_path = _get_user_specific_config_path()
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_hobbies(filepath: str = 'hobbies.json') -> list[str]:
    """Loads a list of hobbies from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return [] 