from groq import Groq  # Import the Groq API
import os
import logging
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_file, render_template, redirect, flash, url_for
from flask_cors import CORS

import sqlite3

from ultralytics import YOLOv10
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Flask app initialization
app = Flask(__name__)
CORS(app)
app.secret_key = 'booomm'

# Setup logging
logging.basicConfig(level=logging.INFO)

# Paths
UPLOAD_FOLDER = 'static/uploads/'
OUTPUT_FOLDER = 'static/outputs/'
MODEL_PATH = 'weights/best.pt'

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Verify YOLOv10 model path
if not os.path.exists(MODEL_PATH):
    logging.error(f"Model path {MODEL_PATH} does not exist.")
else:
    logging.info(f"Model path {MODEL_PATH} exists.")

# Initialize YOLOv10 model
model = YOLOv10(MODEL_PATH)

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_ENV")

if not PINECONE_API_KEY or not PINECONE_API_ENV:
    logging.error("Pinecone API key or environment not set.")
    raise ValueError("Pinecone API key or environment not set.")


# Initialize the Groq API
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def query_deepseek(messages, temperature=0.6, max_tokens=4096, top_p=0.95):
    try:
        # Send the request to the Groq API
        completion = client.chat.completions.create(
            model="deepseek-r1-distill-qwen-32b",
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=top_p,
        )
        # Extract the message content from the response
        if completion.choices and len(completion.choices) > 0:
            return completion.choices[0].message.content
        raise ValueError("Unexpected response from Groq API.")
    except Exception as e:
        logging.error(f"Error querying DeepSeek: {e}")
        return "Sorry, there was an error processing your request."


# User database (replace with proper database in production)
users = {"username": "password"}

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in users and users[username] == password:
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials, please try again.', 'danger')
            return render_template('login.html')
    return render_template('login.html')

# Signup route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username not in users:
            users[username] = password
            flash('Account created successfully!', 'success')
            return redirect(url_for('login'))
        else:
            flash('Username already exists!', 'danger')
            return render_template('signup.html')
    
    return render_template('signup.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_patient', methods=['GET', 'POST'])
def add_patient():
    if request.method == 'POST':
        # ... (code for adding patient information)
        return "Patient information added successfully!"
    return render_template('add_patient.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        user_input = request.json.get("message", "").strip()

        if "patient" in user_input.lower():
            # Patient query handled by SQLite database
            patient_name = user_input.lower().split("patient")[-1].strip()
            conn = sqlite3.connect("patients_info.db")
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM patients WHERE name LIKE ?", ('%' + patient_name + '%',))
            patient_info = cursor.fetchall()
            conn.close()

            if patient_info:#
                response = "Patient Information:\n"
                for patient in patient_info:
                    response += (
                        f"Name: {patient[1]}\n"
                        f"Date of Birth: {patient[2]}\n"
                        f"Gender: {patient[3]}\n"
                        f"Medical History: {patient[4]}\n"
                        f"Tumor Type: {patient[5]}\n"
                        f"Tumor Location: {patient[6]}\n"
                        f"Tumor Size: {patient[7]}\n"
                        f"Diagnosis Date: {patient[8]}\n"
                        f"Treatment Plan: {patient[9]}\n"
                        f"Follow-Up Reports: {patient[10]}\n"
                        f"Treating Physician: {patient[11]}\n"
                        f"Next Appointment: {patient[12]}\n"
                        f"Contact Info: {patient[13]}\n"
                        f"Additional Notes: {patient[14]}\n\n"
                    )
                return jsonify({"response": response.strip()})
            else:
                return jsonify({"response": "No patient found with the given details."})
        else:
            # Attempt DeepSeek query
            try:
                messages = [{"role": "user", "content": user_input}]
                deepseek_response = query_deepseek(messages)
                if "Sorry" in deepseek_response:  # Check if DeepSeek fails
                    raise Exception("DeepSeek failed.")
                return jsonify({"response": deepseek_response})
            except Exception:
                # Fallback to OpenAI GPT-4o-mini
                fallback_prompt = f"Your Name is Jaggu, you are medical assistant, Just give the Answer in paragrapgh and in 2 3 lines Please assist with the following query: {user_input}"
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    store=True,
                    messages=[{"role": "user", "content": fallback_prompt}]
                )
                openai_response = completion.choices[0].message.content
                return jsonify({"response": openai_response})

    except Exception as e:
        logging.error(f"Error in chatbot route: {e}")
        return jsonify({"error": str(e)}), 500

from openai import OpenAI
import os
import logging
import uuid
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, url_for



from dotenv import load_dotenv
import os
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, str(uuid.uuid4()) + "_" + file.filename)
        file.save(file_path)

        try:
            # YOLOv10 prediction
            result = model.predict(source=file_path, imgsz=640, conf=0.25)
            if isinstance(result, list) and len(result) > 0:
                result = result[0]

                # Extract YOLO annotations
                predictions = []
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        predictions.append({
                            "class": result.names[box.cls[0].item()],
                            "confidence": box.conf[0].item(),
                            "bbox": box.xyxy[0].tolist()
                        })

                yolo_summary = "\n".join(
                    f"Detected {pred['class']} with confidence {pred['confidence']:.2f}."
                    for pred in predictions
                )

                # Generate AI interpretation using GPT-4o-mini
                prompt = f"""
                I have an MRI image of the brain, and the following findings were detected:
                {yolo_summary}

                Can you provide a detailed explanation of what these detections mean? Include possible tumor types, affected areas, and recommendations for diagnosis.in 2 to 3 line 
                """

                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    store=True,
                    messages=[{"role": "user", "content": prompt}]
                )
                ai_analysis = completion.choices[0].message.content

                # Save annotated image
                if hasattr(result, 'plot'):
                    annotated_img = result.plot()
                    output_path = os.path.join(OUTPUT_FOLDER, os.path.basename(file_path))
                    plt.imsave(output_path, annotated_img[:, :, ::-1])

                    return jsonify({
                        "annotated_image": url_for('static', filename=f"outputs/{os.path.basename(output_path)}"),
                        "yolo_summary": yolo_summary,
                        "ai_analysis": ai_analysis
                    })

            return jsonify({"error": "Failed to process the image"}), 500

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return jsonify({"error": str(e)}), 500




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)
