import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import ollama

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure Ollama API Host and Model
OLLAMA_API_HOST = os.getenv('OLLAMA_API_HOST', "http://localhost:11434")
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama2')

@app.route('/explain_moves', methods=['POST'])
def explain_moves():
    """
    Explains a sequence of chess moves from a given FEN position.
    Expects a JSON body with 'fen' (string) and 'move_sequence' (list of strings).
    """
    try:
        data = request.get_json()
        fen = data.get('fen')
        move_sequence = data.get('move_sequence')

        if not fen or not move_sequence:
            return jsonify({'error': 'Missing FEN or move_sequence in request body'}), 400

        if not isinstance(fen, str):
            return jsonify({'error': 'FEN must be a string'}), 400
        if not isinstance(move_sequence, list) or not all(isinstance(move, str) for move in move_sequence):
            return jsonify({'error': 'move_sequence must be a list of strings'}), 400

        # Initialize Ollama client
        try:
            client = ollama.Client(host=OLLAMA_API_HOST)
        except Exception as e:
            print(f"Error initializing Ollama client: {e}")
            return jsonify({'error': f'Failed to connect to Ollama: {str(e)}'}), 500

        moves_str = " ".join(move_sequence)

        prompt_text = f"""
You are a chess assistant.
Given the chess position represented by the FEN string:
{fen}

Explain the strategic idea, tactical motifs, and potential consequences behind the following sequence of moves:
{moves_str}

Provide a concise explanation focusing on the plan for the side making these moves.
"""

        try:
            response = client.chat(
                model=OLLAMA_MODEL,
                messages=[{'role': 'user', 'content': prompt_text}]
            )
            explanation = response['message']['content']
        except Exception as e:
            print(f"Error during Ollama API call: {e}")
            # Attempt to see if the model exists if a model not found error is suspected
            try:
                client.list() # check connection
            except Exception as client_e:
                 return jsonify({'error': f'Ollama client error: {str(client_e)}. Is OLLAMA_API_HOST ({OLLAMA_API_HOST}) correct and Ollama running?'}), 500
            
            available_models = [m['name'] for m in client.list()['models']]
            if f"{OLLAMA_MODEL}:latest" not in available_models and OLLAMA_MODEL not in available_models : # ollama client appends :latest automatically
                 return jsonify({'error': f'Ollama model "{OLLAMA_MODEL}" not found. Available models: {", ".join(available_models)}. Or connection issue.'}), 500
            return jsonify({'error': f'Error communicating with Ollama model: {str(e)}'}), 500


        return jsonify({'explanation': explanation})

    except Exception as e:
        # Catch-all for any other unexpected errors
        print(f"Unexpected error in /explain_moves: {e}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    # Make sure to set OLLAMA_API_HOST and OLLAMA_MODEL environment variables if needed.
    print(f"Starting Flask app. Ollama host: {OLLAMA_API_HOST}, Model: {OLLAMA_MODEL}")
    app.run(debug=True, port=5000)
