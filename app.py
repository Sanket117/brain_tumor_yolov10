import os
import logging
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_file, render_template, redirect, flash, url_for
from flask_cors import CORS
from langchain_pinecone import Pinecone as LangChainPinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
import sqlite3
from pinecone import Pinecone
from ultralytics import YOLOv10
import uuid

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
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")

if not PINECONE_API_KEY or not PINECONE_API_ENV:
    logging.error("Pinecone API key or environment not set.")
    raise ValueError("Pinecone API key or environment not set.")

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalchatbot"

if index_name not in pc.list_indexes().names():
    logging.error(f"Index {index_name} does not exist.")
    raise ValueError(f"Index {index_name} does not exist.")

index = pc.Index(index_name)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Initialize Pinecone vector store
docsearch = LangChainPinecone(index=index, embedding=embeddings, text_key='text')

# Define the prompt template
prompt_template = """
Use the following pieces of medical context to answer the user's question.
If you don't know the answer, just say that you don't know; don't try to make up an answer.

Context: {context}
Question: {question}

Provide a concise, professional answer based strictly on the context.
Answer:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# Initialize DeepSeek R1 through Ollama
llm = Ollama(
    model="deepseek-r1:8b",
    temperature=0.7,
    num_predict=512,
    system="You're a medical assistant. Answer strictly based on context."
)

# Create the RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

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
        # ... (same patient registration code as before)
        return "Patient information added successfully!"
    return render_template('add_patient.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        user_input = request.json['message']
        
        if "patient" in user_input.lower():
            patient_name = user_input.lower().split("patient")[-1].strip()
            conn = sqlite3.connect('patients_info.db')
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM patients WHERE name LIKE ?", ('%' + patient_name + '%',))
            patient_info = cursor.fetchall()
            conn.close()

            if patient_info:
                response = "Patient Information:\n"
                for patient in patient_info:
                    response += (f"Name: {patient[1]}\n"
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
            return jsonify({"response": "Patient not found."})

        result = qa({"query": user_input})
        response = result["result"]
        return jsonify({"response": response})

    except Exception as e:
        logging.error(f"Error during chatbot prediction: {e}")
        return jsonify({"error": str(e)}), 500

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
            result = model.predict(source=file_path, imgsz=640, conf=0.25)
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
                if hasattr(result, 'plot'):
                    annotated_img = result.plot()
                    output_path = os.path.join(OUTPUT_FOLDER, os.path.basename(file_path))
                    plt.imsave(output_path, annotated_img[:, :, ::-1])
                    return send_file(output_path, mimetype='image/jpeg')
            return jsonify({"error": "Failed to process the image"}), 500

        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)