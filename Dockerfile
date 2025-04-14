FROM tensorflow/serving

# Expose the port
EXPOSE 8080

# Start TensorFlow Serving with the correct syntax
ENTRYPOINT ["tensorflow_model_server", "--port=8080", "--rest_api_port=8080", "--model_name=spam_detection_model", "--model_base_path=/models/spam_detection_model"]
