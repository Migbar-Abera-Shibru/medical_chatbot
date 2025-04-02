import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import groq
from pinecone import Pinecone, ServerlessSpec  # Updated import
from PyPDF2 import PdfReader
from docx import Document
from unstructured.partition.auto import partition
from sentence_transformers import SentenceTransformer
import numpy as np
import tempfile
import shutil
from typing import List, Dict, Any
import uuid
import traceback
from datetime import datetime

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx', 'md'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Groq client
try:
    groq_client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    print("Failed to initialize Groq client:", str(e))
    groq_client = None



# Initialize Sentence Transformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Supported Models
SUPPORTED_MODELS = {
    "fast": "llama3-8b-8192",
    "powerful": "llama3-70b-8192",
    "long-context": "llama-3.3-70b-versatile"
}

def initialize_pinecone():
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Verify connection
        if not pc.list_indexes():
            raise Exception("Failed to connect to Pinecone API")
        
        pinecone_index_name = "chatbot-docs"
        
        # Create index if needed
        if pinecone_index_name not in pc.list_indexes().names():
            print("Creating new index...")
            pc.create_index(
                name=pinecone_index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            time.sleep(60)  # Wait for index to initialize
        
        # Get index with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                index = pc.Index(pinecone_index_name)
                # Verify index is ready
                index.describe_index_stats()
                return index
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = 2 ** (attempt + 1)
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                
    except Exception as e:
        print(f"Pinecone initialization failed: {str(e)}")
        return None

# Initialize at startup
pinecone_index = initialize_pinecone()
if not pinecone_index:
    print("WARNING: Pinecone not available - document uploads will fail")

# Add to your add_to_pinecone function:
def add_to_pinecone(chunks, embeddings, filename):
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            'id': f"{filename}-{i}",
            'values': embedding.tolist(),
            'metadata': {
                'text': chunk[:1000],
                'filename': filename
            }
        })
        
        # Upload in smaller batches with timeout
        if len(vectors) >= 20:  # Reduced batch size
            try:
                pinecone_index.upsert(
                    vectors=vectors,
                    timeout=30  # 30 second timeout
                )
                vectors = []
            except Exception as e:
                print(f"Batch upload failed, retrying... Error: {str(e)}")
                time.sleep(5)
                try:
                    pinecone_index.upsert(vectors=vectors)
                    vectors = []
                except Exception as e:
                    raise Exception(f"Final upload failed: {str(e)}")
    
    # Upload remaining vectors
    if vectors:
        pinecone_index.upsert(vectors=vectors)
    
    return len(chunks)

def query_pinecone(query_embedding: np.ndarray, top_k: int = 3) -> str:
    """Query Pinecone for relevant chunks"""
    if not pinecone_index:
        return ""
    
    results = pinecone_index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )
    
    context = ""
    for match in results.matches:
        context += f"\n\nDocument excerpt:\n{match.metadata['text']}"
        context += f"\n(Source: {match.metadata['filename']})"
    
    return context

@app.route('/')
def home():
    return jsonify({
        "status": "API is running",
        "supported_models": SUPPORTED_MODELS,
        "vector_db": "pinecone" if pinecone_index else "none",
        "endpoints": {
            "chat": "/api/chat (POST)",
            "upload": "/api/upload (POST)"
        }
    })

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(file_path: str, file_extension: str) -> str:
    """Extract text from different file formats"""
    text = ""
    try:
        if file_extension == 'pdf':
            reader = PdfReader(file_path)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif file_extension == 'docx':
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs if para.text])
        elif file_extension in ['txt', 'md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            elements = partition(filename=file_path)
            text = "\n".join([str(el) for el in elements])
    except Exception as e:
        print(f"Error extracting text: {e}")
        raise
    return text

def preprocess_text(text: str) -> str:
    """Basic text preprocessing"""
    return ' '.join(text.split())

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 30) -> List[str]:
    """More efficient chunking with smaller sizes"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def add_to_pinecone(chunks: List[str], embeddings: List[np.ndarray], filename: str) -> int:
    """Upload document chunks to Pinecone in batches"""
    batch_size = 50  # Adjust based on your chunk size
    vectors = []
    total_uploaded = 0
    
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            'id': f"{filename}-{i}",
            'values': embedding.tolist(),
            'metadata': {
                'text': chunk[:1000],  # Store first 1000 chars
                'filename': filename,
                'chunk_num': i
            }
        })
        
        # Upload in batches
        if len(vectors) >= batch_size:
            try:
                pinecone_index.upsert(vectors=vectors)
                total_uploaded += len(vectors)
                print(f"Uploaded batch {i//batch_size + 1} ({total_uploaded} total)")
                vectors = []  # Reset batch
            except Exception as e:
                print(f"Failed to upload batch: {str(e)}")
                raise
    
    # Upload remaining vectors
    if vectors:
        pinecone_index.upsert(vectors=vectors)
        total_uploaded += len(vectors)
    
    return total_uploaded

def query_pinecone(query_embedding: np.ndarray, top_k: int = 3) -> str:
    """Query Pinecone for relevant chunks"""
    if not pinecone_index:
        return ""
    
    results = pinecone_index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )
    
    context = ""
    for match in results.matches:
        context += f"\n\nDocument excerpt:\n{match.metadata['text']}"
        context += f"\n(Source: {match.metadata['filename']})"
    
    return context

def generate_embeddings(text_chunks: List[str]) -> List[np.ndarray]:
    """Generate embeddings for text chunks"""
    try:
        print(f"Generating embeddings for {len(text_chunks)} chunks...")
        embeddings = embedding_model.encode(text_chunks)
        print("Embeddings generated successfully")
        return embeddings
    except Exception as e:
        print(f"Embedding generation failed: {str(e)}")
        raise

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if not pinecone_index:
        return jsonify({'error': 'Pinecone service unavailable'}), 503
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    try:
        # Save to temp location
        temp_dir = tempfile.mkdtemp()
        filename = secure_filename(file.filename)
        filepath = os.path.join(temp_dir, filename)
        file.save(filepath)
        
        # Process file
        raw_text = extract_text_from_file(filepath, filename.rsplit('.', 1)[1].lower())
        processed_text = preprocess_text(raw_text)
        chunks = chunk_text(processed_text)
        embeddings = generate_embeddings(chunks)
        
        # Upload to Pinecone
        chunk_count = add_to_pinecone(chunks, embeddings, filename)
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        return jsonify({
            'message': f'File "{filename}" processed successfully!',
            'chunk_count': chunk_count,
            'status': 'ready_for_query'
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '')
        model_type = data.get('model_type', 'fast')
        
        if not message:
            return jsonify({'error': 'Empty message'}), 400
        
        if not groq_client:
            return jsonify({'error': 'Groq client not initialized'}), 500

        model = SUPPORTED_MODELS.get(model_type, "llama3-8b-8192")
        
        # Get relevant context from Pinecone
        query_embedding = embedding_model.encode([message])[0]
        context = query_pinecone(query_embedding) if pinecone_index else ""

        # Call Groq API
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant." + 
                              (context if context else "\nUse general knowledge to answer.")
                },
                {
                    "role": "user",
                    "content": message
                }
            ],
            model=model,
            temperature=0.5,
            max_tokens=1024
        )
        
        return jsonify({
            'response': chat_completion.choices[0].message.content,
            'model_used': model,
            'context_used': bool(context)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'supported_models': list(SUPPORTED_MODELS.values())
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)