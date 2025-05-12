import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from pathlib import Path

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
    mask_expanded = np.expand_dims(attention_mask, axis=-1)
    sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
    sum_mask = np.sum(attention_mask, axis=1, keepdims=True)
    sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=None)  # Avoid division by zero
    return sum_embeddings / sum_mask

# Sentences we want sentence embeddings for
sentences = ['This is an example sentence', 'Each sentence is converted']

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model_path = Path('./models/all-MiniLM-L6-v2.onnx')
ort_session = ort.InferenceSession(str(model_path))

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='np')

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

# Normalize embeddings
sentence_embeddings = sentence_embeddings / np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)

print("Sentence embeddings:")
print(sentence_embeddings)