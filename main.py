from flask import Flask, request, send_from_directory
import numpy as np
import time
import os
import onnxruntime as ort
from transformers import AutoTokenizer
from pathlib import Path
from huggingface_hub import hf_hub_download

# Initialize Flask app with static files from the root directory
app = Flask(__name__, static_url_path='', static_folder='.')

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
    mask_expanded = np.expand_dims(attention_mask, axis=-1)
    sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
    sum_mask = np.sum(attention_mask, axis=1, keepdims=True)
    sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=None)  # Avoid division by zero
    return sum_embeddings / sum_mask

# Function to download model if it doesn't exist
def download_model():
    model_dir = Path('./models')
    model_path = model_dir / 'all-MiniLM-L6-v2.onnx'

    # Create models directory if it doesn't exist
    if not model_dir.exists():
        print("Creating models directory...")
        model_dir.mkdir(parents=True, exist_ok=True)

    # Check if model already exists
    if model_path.exists():
        print(f"Model already exists at {model_path}")
        return model_path

    print("Model not found locally. Downloading from Hugging Face Hub...")
    try:
        # Download the model from Hugging Face Hub
        downloaded_path = hf_hub_download(
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
            filename="onnx/model.onnx",
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )

        # Rename the downloaded file to match expected name
        os.rename(downloaded_path, model_path)
        print(f"Model downloaded successfully to {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise

# Load the model when the server starts
print("Loading model... This may take a moment.")
try:
    # Load tokenizer from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    # Download model if it doesn't exist locally
    model_path = download_model()

    # Load the ONNX model
    ort_session = ort.InferenceSession(str(model_path))
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please make sure you have installed all dependencies with: pip install -r requirements.txt")
    exit(1)

# FAQ data
FAQS = [
    {
        "question": "Who reviews my application?",
        "answer": "Our admin team and NGO partners carefully verify every application."
    },
    {
        "question": "Is there any helpline I can call?",
        "answer": "Yes! You can contact our support team for further assistance. You can find our contact details by following these steps: Click on the three horizontal lines (☰) at the top left corner of the homepage. Select 'Contact Us' from the menu."
    },
    {
        "question": "What if I don't have an NID?",
        "answer": "You can use a birth certificate or guardian ID with proper explanation."
    },
    {
        "question": "How can I make a donation",
        "answer": "You can donate through our app using mobile banking, card, or manual bank transfer. Just click \"Donate Now\" on the home screen , login/signup and follow the steps!"
    },
    {
        "question": "What if my application is rejected?",
        "answer": "You'll receive a message explaining why. You can reapply with updated info."
    },
    {
        "question": "How long does it take to get help?",
        "answer": "If your request is verified, aid is usually sent within 2–3 days."
    },
    {
        "question": "How do you ensure the right people get the money?",
        "answer": "All relief applicants go through a strict verification process before being approved."
    },
    {
        "question": "Is there a minimum amount I can donate?",
        "answer": "Yes, the minimum donation is BDT 50. Every little bit counts!"
    },
    {
        "question": "How can I apply for financial help?",
        "answer": "Go to the \"Apply for Relief\" section, and fill out the form with your correct details."
    },
    {
        "question": "Can I apply on behalf of someone else?",
        "answer": "Yes, with their consent and proper documentation."
    }
]

# Cache for FAQ embeddings
FAQ_EMBEDDINGS = None

# Helper function to convert numpy types to Python native types
def convert_numpy_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj

# Function to encode text using ONNX model
def get_embedding(texts, normalize_embeddings=True):
    # Ensure texts is a list
    if isinstance(texts, str):
        texts = [texts]

    # Tokenize sentences
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='np')

    # Prepare inputs (convert to int64 as required by the ONNX model)
    model_inputs = {
        'input_ids': encoded_input['input_ids'].astype(np.int64),
        'attention_mask': encoded_input['attention_mask'].astype(np.int64)
    }

    # Add token_type_ids if needed by the model
    if 'token_type_ids' in [input.name for input in ort_session.get_inputs()]:
        model_inputs['token_type_ids'] = encoded_input.get('token_type_ids',
                                         np.zeros_like(encoded_input['input_ids'])).astype(np.int64)

    # Run inference
    outputs = ort_session.run(None, model_inputs)

    # Perform pooling
    sentence_embeddings = mean_pooling(outputs[0], encoded_input['attention_mask'])

    # Normalize embeddings if requested
    if normalize_embeddings:
        sentence_embeddings = sentence_embeddings / np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)

    return sentence_embeddings

# Function to calculate cosine similarity
def cosine_similarity(a, b):
    # If a is a single embedding and b is a matrix of embeddings
    if len(a.shape) == 1:
        a = a.reshape(1, -1)

    # Calculate dot product
    dot_product = np.dot(a, b.T)

    # Calculate norms
    norm_a = np.linalg.norm(a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(b, axis=1, keepdims=True)

    # Calculate cosine similarity
    return dot_product / (norm_a * norm_b.T)

# Function to calculate cosine similarity between query and FAQs
def find_best_faq_match(query_text: str, top_k: int = 1):
    global FAQ_EMBEDDINGS

    # Ensure top_k is positive
    top_k = max(1, top_k)

    # Encode the query
    query_embedding = get_embedding(query_text)

    # Encode FAQs if not already cached
    if FAQ_EMBEDDINGS is None:
        faq_questions = [faq["question"] for faq in FAQS]
        FAQ_EMBEDDINGS = get_embedding(faq_questions)

    # Calculate cosine similarities
    cos_scores = cosine_similarity(query_embedding, FAQ_EMBEDDINGS)[0]

    # Get top-k matches
    top_results = []

    # Convert to numpy array if it's not already
    if not isinstance(cos_scores, np.ndarray):
        cos_scores = np.array(cos_scores)

    # Get indices of top-k scores
    top_indices = np.argsort(cos_scores)
    top_indices = top_indices[-top_k:]  # Take last k elements (highest scores)
    top_indices = top_indices[::-1]     # Reverse to get descending order

    for idx in top_indices:
        top_results.append({
            "question": FAQS[int(idx)]["question"],
            "answer": FAQS[int(idx)]["answer"],
            "score": float(cos_scores[idx])
        })

    return top_results

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/embed', methods=['POST'])
def embed():
    start_time = time.time()

    # Get text from request
    data = request.get_json()

    if not data or 'texts' not in data:
        return {'error': 'Please provide texts to embed'}, 400

    texts = data['texts']

    # Validate input
    if not isinstance(texts, list):
        texts = [texts]  # Convert single text to list

    # Generate embeddings
    try:
        embeddings = get_embedding(texts)
        processing_time = time.time() - start_time

        # Convert numpy types to Python native types
        embeddings_list = [convert_numpy_types(embedding) for embedding in embeddings]
        dimensions = int(embeddings.shape[1])
        processing_time = float(processing_time)

        return {
            'embeddings': embeddings_list,
            'dimensions': dimensions,
            'processing_time_seconds': processing_time,
            'texts_processed': len(texts)
        }

    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/health', methods=['GET'])
def health():
    return {
        'status': 'ok',
        'message': 'Server is running'
    }

@app.route('/info', methods=['GET'])
def model_info():
    # Get model inputs to determine embedding dimensions
    sample_embedding = get_embedding(["Sample text for dimension check"])
    embedding_dimensions = sample_embedding.shape[1]

    # Get max sequence length from tokenizer
    max_seq_length = tokenizer.model_max_length

    return {
        'model_name': 'sentence-transformers/all-MiniLM-L6-v2 (ONNX)',
        'embedding_dimensions': int(embedding_dimensions),
        'max_sequence_length': int(max_seq_length)
    }

@app.route('/faq', methods=['POST'])
def faq_query():
    start_time = time.time()

    # Get query from request
    data = request.get_json()

    if not data or 'query' not in data:
        return {'error': 'Please provide a query'}, 400

    query = data['query']
    top_k = data.get('top_k', 1)  # Default to 1 if not provided

    # Validate input
    if not isinstance(query, str):
        return {'error': 'Query must be a string'}, 400

    if not isinstance(top_k, int) or top_k < 1:
        return {'error': 'top_k must be a positive integer'}, 400

    # Find best matching FAQ
    try:
        matches = find_best_faq_match(query, top_k)
        processing_time = time.time() - start_time

        return {
            'matches': matches,
            'processing_time_seconds': float(processing_time),
            'query': query
        }

    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/answer', methods=['POST'])
def answer():
    # Get query from request
    data = request.get_json()

    if not data or 'question' not in data:
        return {'error': 'Please provide a question'}, 400

    question = data['question']

    # Validate input
    if not isinstance(question, str):
        return {'error': 'Question must be a string'}, 400

    # Find best matching FAQ
    try:
        matches = find_best_faq_match(question, 1)

        if matches and len(matches) > 0 and matches[0].get('score', 0) > 0.5:
            match = matches[0]
            return {
                'question': question,
                'answer': match['answer'],
                'matchedQuestion': match['question'],
                'confidence': float(match['score'])
            }
        else:
            return {
                'question': question,
                'answer': 'Sorry, I could not find an answer to your question.',
                'matchedQuestion': '',
                'confidence': 0.0
            }

    except Exception as e:
        return {'error': str(e)}, 500

if __name__ == '__main__':
    try:
        # Get port from environment variable or default to 5000
        import os
        port = int(os.environ.get('PORT', 5000))

        print(f"Starting server on port {port}")
        print("Press Ctrl+C to stop the server")
        app.run(debug=True, host='0.0.0.0', port=port)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"\nError starting server: {e}")
