from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from ultralytics import YOLOv10
import matplotlib.pyplot as plt
import os
import uuid
import logging
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the tokenizer and model
tokenizer_chatbot = GPT2Tokenizer.from_pretrained('gpt2')
model_chatbot = GPT2LMHeadModel.from_pretrained('gpt2')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_chatbot = model_chatbot.to(device)

app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Paths
UPLOAD_FOLDER = 'static/uploads/'
OUTPUT_FOLDER = 'static/outputs/'
MODEL_PATH = 'weights/best.pt'

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Verify model path
if not os.path.exists(MODEL_PATH):
    logging.error(f"Model path {MODEL_PATH} does not exist.")
else:
    logging.info(f"Model path {MODEL_PATH} exists.")

# Load the model
model = YOLOv10(MODEL_PATH)

# Serve the index.html file
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        user_input = request.json['message']
        
        # Define generation parameters
        max_length = 200
        min_length = 20
        temperature = 0.7
        top_k = 50
        
        # Tokenize the input
        input_ids = tokenizer_chatbot.encode(user_input, return_tensors='pt')
        
        # Generate response
        with torch.no_grad():
            input_ids = input_ids.to(device)
            chatbot_output = model_chatbot.generate(
                input_ids,
                max_length=max_length,
                min_length=min_length,
                temperature=temperature,
                top_k=top_k,
                num_beams=1
            )
        
        # Decode and return response
        response = tokenizer_chatbot.decode(chatbot_output[0], skip_special_tokens=True)
        response = filter_response(response)
        
        return jsonify({"response": response})
    
    except Exception as e:
        logging.error(f"Error during chatbot prediction: {e}")
        return jsonify({"error": str(e)}), 500

def filter_response(response):
    parts = response.split('.')
    unique_parts = list(set(parts))
    cleaned_response = '.'.join(unique_parts)
    return cleaned_response

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, str(uuid.uuid4()) + "_" + file.filename)
        logging.info(f"Saving uploaded file to: {file_path}")
        file.save(file_path)

        logging.info(f"Predicting with model: {file_path}")
        try:
            result = model.predict(source=file_path, imgsz=640, conf=0.25)
            logging.info(f"Prediction result: {result}")

            if isinstance(result, list) and len(result) > 0:
                result = result[0]
                if hasattr(result, 'plot'):
                    annotated_img = result.plot()
                    output_path = os.path.join(OUTPUT_FOLDER, os.path.basename(file_path))
                    logging.info(f"Saving annotated image to: {output_path}")
                    plt.imsave(output_path, annotated_img[:, :, ::-1])
                    return send_file(output_path, mimetype='image/jpeg')
                else:
                    logging.error(f"Unexpected result structure: {result}")
            else:
                logging.error(f"Unexpected result format: {result}")

            return jsonify({"error": "Failed to process the image"}), 500

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
