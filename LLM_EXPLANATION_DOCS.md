## New Feature: LLM Move Explanation

This feature allows users to get an AI-generated explanation for the strategic ideas and tactical motifs behind Stockfish's recommended move sequences (principal variations). It integrates with a local Large Language Model (LLM) via Ollama to provide these insights.

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
    *   If you prefer to use a different model, you can specify it using the `OLLAMA_MODEL` environment variable when running the backend (see below). For example, to use `mistral`, you would run `ollama pull mistral`.

**2. Python Backend Setup:**

*   **Python Environment:**
    *   Ensure you have Python installed (Python 3.8+ is recommended).
*   **Install Dependencies:**
    *   The backend server requires the following Python libraries: `Flask`, `ollama`, and `flask-cors`.
    *   Install them using pip:
        ```bash
        pip install Flask ollama flask-cors
        ```
*   **Run the Backend Server:**
    *   Navigate to the directory containing the `ollama_explainer.py` file in your terminal.
    *   Run the server using:
        ```bash
        python ollama_explainer.py
        ```
    *   By default, the server will start on `http://localhost:5000`. You should see output in your terminal indicating the server is running, similar to:
        ```
        * Serving Flask app 'ollama_explainer'
        * Debug mode: on
        WARNING: This is a development server. Do not use it in a production deployment.
        Use a production WSGI server instead.
        * Running on http://127.0.0.1:5000
        Press CTRL+C to quit
        Starting Flask app. Ollama host: http://localhost:11434, Model: llama2
        ```
*   **Environment Variables (Optional Configuration):**
    *   You can configure the backend server using environment variables:
        *   `OLLAMA_API_HOST`: Specifies the URL of your Ollama API. This is useful if Ollama is running on a different machine or port.
            *   Defaults to: `http://localhost:11434`
            *   Example:
                ```bash
                export OLLAMA_API_HOST="http://192.168.1.10:11434" 
                # On Windows, use: set OLLAMA_API_HOST="http://192.168.1.10:11434"
                python ollama_explainer.py 
                ```
        *   `OLLAMA_MODEL`: Specifies the name of the Ollama model you want to use for explanations. Make sure you have pulled this model using `ollama pull <model_name>`.
            *   Defaults to: `llama2`
            *   Example:
                ```bash
                export OLLAMA_MODEL="mistral"
                # On Windows, use: set OLLAMA_MODEL="mistral"
                python ollama_explainer.py
                ```

### How to Use

1.  **Start Services:**
    *   Ensure the Ollama service is running and the desired LLM (e.g., `llama2`) has been pulled.
    *   Start the Python backend server (`python ollama_explainer.py`).
2.  **Open Application:**
    *   Open the `index.html` file in your web browser.
3.  **Get Best Moves:**
    *   Use the "Show Best" button. Stockfish will analyze the current position and display its top recommended move(s). The best move will be highlighted on the board.
4.  **Explain Plan:**
    *   Once a best move is highlighted (and its sequence/plan is loaded from Stockfish), click the "Explain Plan" button.
5.  **View Explanation:**
    *   The application will send the current position (FEN) and the move sequence to the Python backend, which then queries the LLM via Ollama.
    *   The AI-generated explanation for the selected plan will appear in the status message area at the bottom of the screen (e.g., "AI Explanation: The plan is to...").

Enjoy exploring the strategic insights provided by the LLM!
