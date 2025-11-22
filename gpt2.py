import json
import numpy as np
from transformer_lens import HookedTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

model = HookedTransformer.from_pretrained("gpt2")

def process_data(data_file):
    """Process prompts from JSON file and extract activation vectors."""
    with open(data_file, "r") as f:
        data = json.load(f)
    
    activation_vectors = []
    labels = []
    
    for entry in data:
        # Process angry prompt (label = 1)
        angry_prompt = entry["angry"]
        tokens = model.to_tokens(angry_prompt)
        _, cache = model.run_with_cache(tokens)
        L = model.cfg.n_layers - 1
        resid_final_preln = cache["resid_post", L]
        vec = resid_final_preln[0, -1]
        activation_vectors.append(vec.cpu().numpy())
        labels.append(1)
        
        # Process neutral prompt (label = 0)
        neutral_prompt = entry["neutral"]
        tokens = model.to_tokens(neutral_prompt)
        _, cache = model.run_with_cache(tokens)
        resid_final_preln = cache["resid_post", L]
        vec = resid_final_preln[0, -1]
        activation_vectors.append(vec.cpu().numpy())
        labels.append(0)
    
    return np.stack(activation_vectors), np.stack(labels)

# Load and process training data
print("Processing training data...")
train_vectors, train_labels = process_data("data/train.json")

# Load and process test data
print("Processing test data...")
test_vectors, test_labels = process_data("data/test.json")

# Save activation vectors and labels
np.save("activation_vectors.npy", train_vectors)
np.save("labels.npy", train_labels)

# Train probe on training data
print("Training probe...")

# I'm doing a simple scikit learn training regime here, but also used
# an MLP in mlp.py
probe = LogisticRegression(max_iter=5000).fit(train_vectors, train_labels)
train_predictions = probe.predict(train_vectors)
train_accuracy = accuracy_score(train_labels, train_predictions)
print(f"Train Accuracy: {train_accuracy}")
train_probabilities = probe.predict_proba(train_vectors)
train_loss = log_loss(train_labels, train_probabilities)
print(f"Train Log Loss: {train_loss}")

# Evaluate on test data
print("Evaluating on test data...")
test_predictions = probe.predict(test_vectors)
test_accuracy = accuracy_score(test_labels, test_predictions)
print(f"Test Accuracy: {test_accuracy}")

# Get predicted probabilities for loss calculation
test_probabilities = probe.predict_proba(test_vectors)
test_loss = log_loss(test_labels, test_probabilities)
print(f"Test Log Loss: {test_loss}")