## New Feature: LLM Move Explanation

This feature allows users to get an AI-generated explanation for the strategic ideas and tactical motifs behind Stockfish's recommended move sequences (principal variations). It integrates with a local Large Language Model (LLM) via Ollama to provide these insights. The move sequences are converted from UCI to SAN (Standard Algebraic Notation) before being sent to the LLM for better readability and potentially more accurate explanations.

### Setup Requirements

To use the LLM Move Explanation feature, you need to set up Ollama and the Python backend server.

**1. Ollama Setup:**

*   **Install Ollama:**
    *   Download and install Ollama from the official website: [https://ollama.com/](https://ollama.com/).
    *   Follow the installation instructions for your operating system.
*   **Ensure Ollama is Running:**
    *   After installation, make sure the Ollama application or service is running. You can typically check this via its command-line interface or system tray icon.
*   **Pull an LLM:**
    *   You need to download an LLM for Ollama to use. The default model for this feature is `llama2`.
    *   Open your terminal or command prompt and run:
        ```bash
        ollama pull llama2
        ```
    *   If you prefer to use a different model (e.g., `mistral`, `codellama:13b`), you can pull it using `ollama pull <model_name>` and then configure it in the `.env` file (see Python Backend Configuration).

**2. Python Backend Setup:**

*   **Python Environment:**
    *   Ensure you have Python installed (Python 3.8+ is recommended).
*   **Install Dependencies:**
    *   The backend server requires the following Python libraries: `Flask`, `ollama`, `flask-cors`, `python-chess` (for UCI to SAN conversion), and `python-dotenv` (for loading configuration from `.env` files).
    *   Install them using pip:
        ```bash
        pip install Flask ollama flask-cors python-chess python-dotenv
        ```
*   **Configuration via `.env` file:**
    *   Configuration for the Python backend (like the Ollama API host and model name) is managed using a `.env` file in the project root.
    *   Copy the example configuration file `.env.example` to a new file named `.env`:
        ```bash
        cp .env.example .env
        ```
    *   Open the `.env` file in a text editor and modify the variables as needed:
        *   `OLLAMA_API_HOST`: Specifies the URL of your Ollama API. This is useful if Ollama is running on a different machine or port.
            *   Default if not set: `http://localhost:11434`
            *   Example: `OLLAMA_API_HOST=http://192.168.1.10:11434`
        *   `OLLAMA_MODEL`: Specifies the name of the Ollama model you want to use for explanations. Make sure you have pulled this model using `ollama pull <model_name>`.
            *   Default if not set: `llama2`
            *   Example: `OLLAMA_MODEL=mistral`
    *   The `.env` file is ignored by Git (as specified in `.gitignore`) and should not be committed to version control, as it may contain environment-specific or sensitive information.
*   **Run the Backend Server:**
    *   Navigate to the directory containing the `ollama_explainer.py` file in your terminal.
    *   Run the server using:
        ```bash
        python ollama_explainer.py
        ```
    *   The server will automatically load settings from your `.env` file.
    *   By default, the server will start on `http://localhost:5000` (accessible from your local machine) or `http://0.0.0.0:5000` (making it accessible from other devices on your network). You should see output in your terminal indicating the server is running, similar to:
        ```
        INFO:werkzeug:Press CTRL+C to quit
        INFO:     root:Starting Flask server on port 5000, OLLAMA_HOST: http://localhost:11434, OLLAMA_MODEL: llama2
        ```
        (The exact logging format might vary slightly based on your Flask and logging setup).

### How to Use

1.  **Start Services:**
    *   Ensure the Ollama service is running and the desired LLM (e.g., `llama2`, or the one configured in your `.env` file) has been pulled.
    *   Start the Python backend server (`python ollama_explainer.py`). It will use the settings from your `.env` file.
2.  **Open Application:**
    *   Open the `index.html` file in your web browser.
3.  **Get Best Moves:**
    *   Use the "Show Best" button. Stockfish will analyze the current position and display its top recommended move(s). The best move will be highlighted on the board.
4.  **Explain Plan:**
    *   Once a best move is highlighted (and its sequence/plan is loaded from Stockfish), click the "Explain Plan" button.
5.  **View Explanation:**
    *   The application will send the current position (FEN) and the move sequence (converted to SAN) to the Python backend, which then queries the LLM via Ollama.
    *   The AI-generated explanation for the selected plan will appear in the status message area at the bottom of the screen (e.g., "AI Explanation: The plan is to...").

Enjoy exploring the strategic insights provided by the LLM!
