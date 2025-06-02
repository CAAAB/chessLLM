import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import ollama

# Configure basic logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure Ollama API Host and Model from environment variables
OLLAMA_API_HOST = os.getenv('OLLAMA_API_HOST', "http://localhost:11434")
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama2')

@app.route('/explain_moves', methods=['POST'])
def explain_moves():
    """
    Explains a sequence of chess moves from a given FEN position.
    Expects a JSON body with 'fen' (string) and 'move_sequence' (list of strings).
    """
    logging.debug("Received request for /explain_moves")
    try:
        data = request.get_json()
        fen = data.get('fen')
        move_sequence = data.get('move_sequence')

        if not fen or not move_sequence:
            logging.warning("Missing FEN or move_sequence in request body")
            return jsonify({'error': 'Missing FEN or move_sequence in request body'}), 400

        if not isinstance(fen, str):
            logging.warning(f"Invalid FEN type: {type(fen)}")
            return jsonify({'error': 'FEN must be a string'}), 400
        if not isinstance(move_sequence, list) or not all(isinstance(move, str) for move in move_sequence):
            logging.warning(f"Invalid move_sequence type or content: {move_sequence}")
            return jsonify({'error': 'move_sequence must be a list of strings'}), 400

        moves_str = " ".join(move_sequence)
        logging.debug(f"FEN: {fen}, Moves: {moves_str}")

        # Initialize Ollama client
        try:
            client = ollama.Client(host=OLLAMA_API_HOST)
        except Exception as e:
            logging.error(f"Error initializing Ollama client with host {OLLAMA_API_HOST}: {e}", exc_info=True)
            return jsonify({'error': f'Failed to connect to Ollama: {str(e)}'}), 500

        prompt_text = f"""
You are a chess assistant.
Given the chess position represented by the FEN string:
{fen}

Explain the strategic idea, tactical motifs, and potential consequences behind the following sequence of moves:
{moves_str}

Provide a concise explanation focusing on the plan for the side making these moves.
"""
        logging.debug(f"Constructed prompt: {prompt_text}")

        try:
            logging.info(f"Sending request to Ollama model: {OLLAMA_MODEL} at host: {OLLAMA_API_HOST}")
            response = client.chat(
                model=OLLAMA_MODEL,
                messages=[{'role': 'user', 'content': prompt_text}]
            )
            logging.debug(f"Ollama response: {response}")
            explanation = response['message']['content']
        except Exception as e:
            logging.error(f"Ollama API call failed: {e}", exc_info=True)
            # Attempt to see if the model exists if a model not found error is suspected
            try:
                client.list() # check connection
            except Exception as client_e:
                 logging.error(f"Ollama client list/connection error: {client_e}", exc_info=True)
                 return jsonify({'error': f'Ollama client error: {str(client_e)}. Is OLLAMA_API_HOST ({OLLAMA_API_HOST}) correct and Ollama running?'}), 500
            
            available_models = []
            try:
                models_info = client.list()
                if models_info and 'models' in models_info:
                    available_models = [m['name'] for m in models_info['models']]
            except Exception as list_exc:
                logging.error(f"Failed to retrieve list of available Ollama models: {list_exc}", exc_info=True)
                # Return a generic error as we can't confirm model availability
                return jsonify({'error': f'Error communicating with Ollama model and failed to list models: {str(e)}'}), 500

            # Check if the model (with and without :latest) is in the list
            model_with_latest = f"{OLLAMA_MODEL}:latest"
            if not (OLLAMA_MODEL in available_models or model_with_latest in available_models):
                 logging.warning(f'Ollama model "{OLLAMA_MODEL}" not found. Available models: {", ".join(available_models)}')
                 return jsonify({'error': f'Ollama model "{OLLAMA_MODEL}" not found. Available models: {", ".join(available_models)}. Or connection issue.'}), 500
            
            # If model seems available, the error was likely something else during chat
            return jsonify({'error': f'Error communicating with Ollama model: {str(e)}'}), 500

        logging.debug("Successfully processed /explain_moves, returning explanation.")
        return jsonify({'explanation': explanation})

    except Exception as e:
        # Catch-all for any other unexpected errors
        logging.error(f"Unexpected error in /explain_moves: {e}", exc_info=True)
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    logging.info(f"Starting Flask server on port 5000, OLLAMA_HOST: {OLLAMA_API_HOST}, OLLAMA_MODEL: {OLLAMA_MODEL}")
    app.run(host='0.0.0.0', port=5000, debug=True)
