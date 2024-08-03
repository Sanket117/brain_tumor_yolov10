import os
import logging
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from langchain_pinecone import Pinecone as LangChainPinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from pinecone import Pinecone

from flask_cors import CORS
from ultralytics import YOLOv10
import matplotlib.pyplot as plt
import os
import uuid
import logging
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel



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

from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Paths
UPLOAD_FOLDER = 'static/uploads/'
OUTPUT_FOLDER = 'static/outputs/'
MODEL_PATH = 'weights/best.pt'
LANGCHAIN_MODEL_PATH = 'model/llama-2-7b-chat.ggmlv3.q4_0.bin'

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Verify YOLOv10 model path
if not os.path.exists(MODEL_PATH):
    logging.error(f"Model path {MODEL_PATH} does not exist.")
else:
    logging.info(f"Model path {MODEL_PATH} exists.")

# Initialize YOLOv10 model
from ultralytics import YOLOv10
model = YOLOv10(MODEL_PATH)

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")

if not PINECONE_API_KEY or not PINECONE_API_ENV:
    logging.error("Pinecone API key or environment not set.")
    raise ValueError("Pinecone API key or environment not set.")

# Create a Pinecone instance
pc = Pinecone(api_key=PINECONE_API_KEY)

# Index name
index_name = "medicalchatbot"

# Check if the index exists
if index_name not in pc.list_indexes().names():
    logging.error(f"Index {index_name} does not exist.")
    raise ValueError(f"Index {index_name} does not exist.")

# Load the existing index
index = pc.Index(index_name)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Initialize Pinecone vector store
docsearch = LangChainPinecone(index=index, embedding=embeddings, text_key='text')

# Define the prompt template
prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know; don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# Initialize the language model
llm = CTransformers(model=LANGCHAIN_MODEL_PATH,
                    model_type="llama",
                    config={'max_new_tokens': 512, 'temperature': 0.8})

# Create the RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

# Serve the index.html file
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        user_input = request.json['message']
        
        # Get response from LangChain-based model
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
