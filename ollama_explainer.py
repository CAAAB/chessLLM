import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import ollama
from ollama import Client, ResponseError # Specific import for ResponseError
import httpx # For specific httpx error handling

# Configure basic logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure Ollama API Host and Model from environment variables
# These will be effectively re-read inside explain_moves if they change, 
# but good for initial log message at startup.
OLLAMA_API_HOST_GLOBAL = os.getenv('OLLAMA_API_HOST', "http://localhost:11434")
OLLAMA_MODEL_GLOBAL = os.getenv('OLLAMA_MODEL', 'llama2')

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

        move_sequence_str = " ".join(move_sequence)
        logging.debug(f"FEN: {fen}, Moves: {move_sequence_str}")

        model_name = os.getenv('OLLAMA_MODEL', 'llama2')
        host = os.getenv('OLLAMA_API_HOST', 'http://localhost:11434')
        
        system_prompt_text = "You are a helpful chess assistant. Your task is to explain the strategic idea, tactical motifs, and potential consequences behind a given sequence of chess moves from a specific board position."

        user_prompt_text = f"""Given the chess position represented by the FEN string:
{fen}

Explain the following sequence of moves:
{move_sequence_str}

Provide a concise explanation focusing on the plan for the side making these moves.
"""
        logging.debug(f"System Prompt: {system_prompt_text}")
        logging.debug(f"User Prompt: {user_prompt_text}")
        
        try:
            # Initialize client here to use potentially updated host from environment
            client = Client(host=host) 
            
            logging.info(f"Sending request to Ollama model: {model_name} at host: {host} with think=False")
            response = client.chat(
                model=model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt_text},
                    {'role': 'user', 'content': user_prompt_text}
                ],
                think=False # Pass think=False as a direct keyword argument as per user's example
            )
            logging.debug(f"Ollama raw response object: {response}")
            
            if response and isinstance(response, dict) and 'message' in response and isinstance(response['message'], dict) and 'content' in response['message']:
                explanation = response['message']['content']
                logging.debug(f"Extracted explanation: {explanation}")
                logging.debug("Successfully processed /explain_moves, returning explanation.")
                return jsonify({'explanation': explanation})
            else:
                logging.error(f"Unexpected Ollama response structure: {response}")
                return jsonify({'error': 'Unexpected response structure from LLM after successful call'}), 500

        except ResponseError as e:
            logging.error(f"Ollama API ResponseError: Status Code: {e.status_code}, Error: {e.error}", exc_info=True)
            return jsonify({'error': f"LLM service ResponseError: {e.status_code} - {str(e.error)}"}), 500
        except httpx.HTTPStatusError as e_httpx_status: 
            logging.error(f"Ollama chat API returned an HTTPStatusError: {e_httpx_status.response.status_code} - {e_httpx_status.response.text}", exc_info=True)
            return jsonify({'error': f"LLM service HTTPStatusError: ({e_httpx_status.response.status_code})."}), 500
        except httpx.RequestError as e_httpx_req: 
            logging.error(f"Could not connect to Ollama service (httpx.RequestError): {e_httpx_req}", exc_info=True)
            return jsonify({'error': "Could not connect to the LLM service (RequestError)."}), 500
        except Exception as e_general: # Catch any other exceptions, including potential client init issues if host is bad
            logging.error(f"An unexpected error occurred while querying Ollama: {e_general}", exc_info=True)
            return jsonify({'error': f"An unexpected error occurred: {str(e_general)}"}), 500

    except Exception as e:
        logging.error(f"Unexpected error in /explain_moves (outer try-except): {e}", exc_info=True)
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    logging.info(f"Starting Flask server on port 5000, OLLAMA_HOST: {OLLAMA_API_HOST_GLOBAL}, OLLAMA_MODEL: {OLLAMA_MODEL_GLOBAL}")
    app.run(host='0.0.0.0', port=5000, debug=True)
