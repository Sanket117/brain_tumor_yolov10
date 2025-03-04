# Brain Tumor Detection and Classification Using YOLOv10 and AI Chatbot Using LLMs

## Overview

This project is a Flask web application that allows users to detect and classify brain tumors using the YOLOv10 model. Additionally, it features an AI chatbot powered by LangChain and LLaMA models to answer medical queries. The application integrates state-of-the-art machine learning and natural language processing technologies.

![Prject ScreenShot](proj_1-1.jpeg)


## Features

- **Brain Tumor Detection**: Detects and classifies brain tumors from MRI images using the YOLOv10 model.
- **AI Chatbot**: Provides answers to medical queries using a LangChain-based chatbot powered by LLaMA and Pinecone.
- **Web Interface**: A simple web interface for uploading images, viewing results, and interacting with the chatbot.
- **Image Annotation**: Annotates detected tumor regions on MRI images and returns the annotated images.

## Project Structure

- **app.py**: The main Flask application file.
- **static/uploads/**: Directory for storing uploaded images.
- **static/outputs/**: Directory for storing annotated output images.
- **static/scripts.js/**: Javascript File.
- **static/style.css/**: CSS file .
- **weights/best.pt**: YOLOv10 model weights.
- **model/llama-2-7b-chat.ggmlv3.q4_0.bin**: LLaMA model file for the AI chatbot.
- **templates/index.html**: HTML file for the main web page.
- **.env/**: Directory for storing API keys.

## Requirements

- Python 3.8 or higher
- Flask
- LangChain
- Pinecone
- Hugging Face Transformers
- Ultralitycs (YOLOv10)
- Matplotlib
- Torch

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Sanket117/brain_tumor_yolov10
   cd brain_tumor_yolov10

2. **Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate