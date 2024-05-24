import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from process_video import ProcessVideo
import faiss
import numpy as np
from config import *

app = Flask(__name__)
video_processor = ProcessVideo()

# Directory to save uploaded videos
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize Faiss index
dimension = DIMENSION
index = faiss.IndexFlatL2(dimension)
vector_store = [] 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('files')
    
    for file in files:
        # Save the file to the uploads directory
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Process the video and get segments and embeddings
        segments, embeddings = video_processor.process(filepath)
        
        for segment, embedding in zip(segments, embeddings):
            vector_data = {
                "filename": file.filename,
                "filepath": filepath,
                "action": segment["action"],
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
                "embedding": embedding["embedding"]
            }
            vector_store.append(vector_data)
            index.add(np.expand_dims(embedding["embedding"], axis=0))
    
    return jsonify({"status": "Files uploaded and processed"})

@app.route('/search', methods=['POST'])
def search_files():
    query = request.json['query']
    # query_embedding = video_processor.encode_text(query)   //modify this function to turn text to encoding using the same text-video model
    
    D, I = index.search(np.array([query_embedding]), k=5)
    
    results = []
    for idx in I[0]:
        result = vector_store[idx]
        results.append(result)
    
    return jsonify(results)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)