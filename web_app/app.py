from flask import Flask, request, jsonify, render_template
import requests
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer and model parameters used for preprocessing
with open('./tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_sequence_length = 100  # Same as the preprocessing during training

app = Flask(__name__)

# Your TensorFlow Serving API URL
API_URL = "https://spam-serving-855965325866.us-central1.run.app/v1/models/spam_detection_model:predict"

@app.route('/')
def home():
    return render_template('index.html')  # HTML form

@app.route('/predict', methods=['POST'])
def predict():
    user_message = request.form['message']

    # Preprocess message
    sequence = tokenizer.texts_to_sequences([user_message])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    payload = {"instances": padded_sequence.tolist()}

    try:
        # Send request to TensorFlow Serving API
        response = requests.post(API_URL, json=payload)

        # Check for non-200 responses
        if response.status_code != 200:
            return jsonify({"error": f"API call failed with status code {response.status_code}"})

        # Attempt to parse JSON response
        prediction = response.json()
        spam_probability = prediction['predictions'][0][0]
        is_spam = spam_probability > 0.5

        # Return result
        result = "Spam" if is_spam else "Not Spam"
        return jsonify({"result": result, "spam_probability": spam_probability})
    except ValueError as e:
        # Handle JSON decoding errors
        return jsonify({"error": "Invalid JSON response from API", "details": str(e)})
    except Exception as e:
        # Handle other exceptions
        return jsonify({"error": "An unexpected error occurred", "details": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
