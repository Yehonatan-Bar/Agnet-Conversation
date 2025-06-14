# Multi-Agent Conversation Orchestrator

This project is a web application that allows you to design, configure, and run conversations between multiple AI agents from different providers (Google Gemini, OpenAI, Anthropic). You can create complex, multi-step workflows where agents collaborate and build upon each other's responses to achieve a final goal.

## Live Demo

You can try a live version of the application here:
[https://agnets-conversation.onrender.com/](https://agnets-conversation.onrender.com/)

*Note: The live demo may require you to provide your own API keys.*

## Features

- **Multi-Agent Workflows**: Define a sequence of AI agents, each with its own persona, instructions, and underlying model.
- **Provider-Agnostic**: Supports models from Google (Gemini), OpenAI (GPT series), and Anthropic (Claude series).
- **Web-Based UI**: An intuitive interface to select a conversation workflow, input your prompt, and watch the conversation unfold in real-time.
- **Dynamic Configuration**: All agent and conversation settings are managed through a `conversationManager.json` file, which can be edited directly.
- **Streaming and Interactivity**: See agent responses as they are generated. Pause and resume the conversation flow.
- **Local First**: Run the entire application on your own machine for privacy and control.

## How to Run Locally

Follow these steps to run the application on your own computer. This gives you full control and ensures your API keys remain private.

### Step 1: Download the Code
- Click the green **Code** button on this repository's page.
- Select **Download ZIP**.

### Step 2: Extract the Folder
- Find the downloaded ZIP file on your computer (usually in your "Downloads" folder).
- Unzip the file. This will create a new folder with the project files.

### Step 3: Install Required Software
You will need Python installed on your computer. If you don't have it, you can download it from [python.org](https://www.python.org/downloads/).

Once Python is installed, you need to install the project's dependencies.

1.  **Open a terminal (or command prompt):**
    *   **Windows**: Open the project folder you extracted, right-click inside the folder while holding the `Shift` key, and select "Open PowerShell window here" or "Open command window here".
    *   **Mac/Linux**: Open the Terminal application, type `cd `, drag the project folder into the terminal window, and press `Enter`.
2.  **Install dependencies:** In the terminal, run the following command:
    ```bash
    pip install -r requirements.txt
    ```

### Step 4: Run the Application
1.  In the same terminal, run this command:
    ```bash
    python conversation_manager.py
    ```
2.  You should see output indicating the server is running, something like:
    `* Running on http://127.0.0.1:5001`

### Step 5: Open in Browser
1.  Open your web browser (like Chrome, Firefox, or Edge).
2.  Go to the address `http://127.0.0.1:5001`.

### Step 6: Enter API Keys
- The application needs API keys to connect to the AI services (Google Gemini, OpenAI, etc.).
- You will see input fields in the UI to enter your API keys.
- Your keys are used by the local server running on your machine to communicate with the AI providers. They are not stored permanently or shared with anyone else. 

**Note:** As an alternative to entering keys in the UI, you can set them as environment variables. The application will automatically use them if no key is provided in the UI. The variable names are `GEMINI_API_KEY`, `OPENAI_API_KEY`, and `ANTHROPIC_API_KEY`. 