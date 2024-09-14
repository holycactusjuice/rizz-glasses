from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods={'GET'})
def hello():
    return "Hello, Flask Server!\n"

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        data = request.get_json()  # Expecting JSON data
        transcription_text = data.get('text', '')  # Extract 'text' from JSON payload
        print(f"Received transcription: {transcription_text}")
        # You can add additional processing here if needed
        response = {"status": "success", "received_text": transcription_text}
    except Exception as e:
        print(f"Error: {e}")
        response = {"status": "error", "message": str(e)}

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Adjust the port as needed