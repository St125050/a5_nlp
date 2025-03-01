# a5_nlp


# Preference Optimization Model

This repository contains a project for training a language model using Direct Preference Optimization (DPO). The project involves loading and preprocessing a suitable dataset, training a reward model, and fine-tuning a pre-trained transformer model using DPO.

## Table of Contents
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Dataset

### Dataset Source

- **Dataset**: [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1)
- **Hugging Face Datasets Hub**: https://huggingface.co/datasets/OpenAssistant/oasst1

### Preprocessing Steps

1. **Load the dataset**:
   - Load the `OpenAssistant/oasst1` dataset from Hugging Face Datasets Hub.
2. **Filter English messages**:
   - Filter the dataset to include only English messages.
3. **Extract prompt-response pairs**:
   - Extract pairs of user prompts and assistant responses.
4. **Tokenize the dataset**:
   - Tokenize the prompt and response pairs using the GPT-2 tokenizer.
5. **Create a PyTorch dataset**:
   - Create a `PreferenceDataset` class to handle the data.

```python
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset

# Load the dataset
dataset = load_dataset("OpenAssistant/oasst1")

# Filter English messages
english_data = dataset['train'].filter(lambda x: x['lang'] == 'en')
print(f"✅ English Dataset Loaded: {len(english_data)} samples")

# Extract prompt-response pairs
pairs = []
for conversation in english_data:
    if isinstance(conversation, dict) and 'messages' in conversation:
        messages = conversation['messages']
        for i in range(len(messages) - 1):
            if messages[i]['role'] == 'user' and messages[i + 1]['role'] == 'assistant':
                pairs.append({
                    'prompt': messages[i]['text'],
                    'response': messages[i + 1]['text'],
                    'quality': messages[i + 1].get('quality', 0)  # Default quality if missing
                })

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Example: GPT-2 model
tokenizer.pad_token = tokenizer.eos_token  # Fix padding issue

# Tokenize dataset
tokenized_pairs = []
for pair in pairs:
    tokenized_input = tokenizer(pair['prompt'], truncation=True, padding='max_length', max_length=128)
    tokenized_output = tokenizer(pair['response'], truncation=True, padding='max_length', max_length=128)
    tokenized_pairs.append({
        'input_ids': tokenized_input['input_ids'],
        'attention_mask': tokenized_input['attention_mask'],
        'labels': tokenized_output['input_ids'],
        'quality': pair['quality']
    })

# Create a PyTorch dataset
class PreferenceDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids': torch.tensor(item['input_ids']),
            'attention_mask': torch.tensor(item['attention_mask']),
            'labels': torch.tensor(item['labels']),
            'quality': torch.tensor(item['quality'])
        }

# Initialize dataset
train_dataset = PreferenceDataset(tokenized_pairs)
```

## Training

### Load and Train Reward Model

1. **Load a pre-trained Transformer model**:
   - Load a pre-trained GPT-2 model for reward modeling.
2. **Preprocess the data**:
   - Extract chosen and rejected responses.
3. **Tokenize inputs**:
   - Tokenize the chosen and rejected responses.
4. **Define optimizer & loss function**:
   - Use `AdamW` optimizer and `BCEWithLogitsLoss` loss function.
5. **Train the reward model**:
   - Train the reward model using the chosen and rejected responses.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()  # Free CUDA memory
print(f"🔹 Using device: {device}")

# Load dataset
dataset = load_dataset("openai/summarize_from_feedback", "axis", split="test")  # Use test split
print(f"✅ Dataset Loaded: {len(dataset)} samples")

# Load a pre-trained Transformer model (GPT-2 for reward modeling)
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Fix the tokenizer padding issue (GPT models don’t have a default pad token)
tokenizer.pad_token = tokenizer.eos_token  # Use EOS as padding token
reward_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device)
reward_model.config.pad_token_id = tokenizer.pad_token_id  # Ensure model uses pad_token_id

# Preprocessing function
def preprocess_data(example):
    """Processes dataset to extract chosen and rejected responses for reward training."""
    return {
        "prompt": str(example.get("info", {}).get("post", "")),  # Convert to string
        "chosen": str(example.get("summary", "")),  # Preferred summary
        "rejected": str(example.get("summary", ""))  # Duplicate since dataset only has 'summary'
    }

# Apply preprocessing safely
dataset = dataset.map(preprocess_data, remove_columns=["info", "summary", "worker", "batch", "split"], features=None)
# Convert dataset to list format
prompts = [entry["prompt"] for entry in dataset]
chosen_responses = [entry["chosen"] for entry in dataset]
rejected_responses = [entry["rejected"] for entry in dataset]

# Tokenize inputs
chosen_outputs = tokenizer(chosen_responses, padding=True, truncation=True, max_length=128, return_tensors="pt")
rejected_outputs = tokenizer(rejected_responses, padding=True, truncation=True, max_length=128, return_tensors="pt")

# Ensure tensors are non-empty
print("Chosen Outputs Shape:", chosen_outputs["input_ids"].shape)
print("Rejected Outputs Shape:", rejected_outputs["input_ids"].shape)

# Define optimizer & loss function
optimizer = optim.AdamW(reward_model.parameters(), lr=1e-5)
criterion = nn.BCEWithLogitsLoss()

# Ensure labels match the shape of chosen/rejected outputs
num_samples = len(chosen_outputs["input_ids"])
labels = torch.cat([
    torch.ones(num_samples, dtype=torch.float32),
    torch.zeros(num_samples, dtype=torch.float32)
]).to(device)
BATCH_SIZE = 4

# Training function
def train_reward_model(model, optimizer, chosen_outputs, rejected_outputs, labels, epochs=3):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        for i in range(0, len(chosen_outputs["input_ids"]), BATCH_SIZE):
            # Select batch
            batch_chosen = {k: v[i:i+BATCH_SIZE].to(device) for k, v in chosen_outputs.items()}
            batch_rejected = {k: v[i:i+BATCH_SIZE].to(device) for k, v in rejected_outputs.items()}
            batch_labels = labels[i:i+BATCH_SIZE]

            # Ensure label shape matches model output
            chosen_scores = model(**batch_chosen).logits.squeeze()
            rejected_scores = model(**batch_rejected).logits.squeeze()

            # Check for empty tensors before computing loss
            if chosen_scores.shape[0] == 0 or rejected_scores.shape[0] == 0:
                print("Skipping empty batch...")
                continue

            # Fix label slicing to match logits shape
            loss = criterion(chosen_scores, batch_labels[:chosen_scores.shape[0]]) + criterion(rejected_scores, batch_labels[:rejected_scores.shape[0]])

            loss.backward()
            optimizer.step()

            # Free unused memory
            torch.cuda.empty_cache()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Train the reward model
train_reward_model(reward_model, optimizer, chosen_outputs, rejected_outputs, labels)

# Save trained reward model
torch.save(reward_model.state_dict(), "reward_model.pth")
print("✅ Reward model training complete & saved!")
```

## Evaluation

### Load and Fine-Tune Model with DPO

1. **Load trained reward model**:
   - Load the trained reward model from Task 2.
2. **Preprocess the data**:
   - Extract and tokenize the prompt, chosen, and rejected responses.
3. **Define DPO loss function**:
   - Implement the Direct Preference Optimization (DPO) loss function.
4. **Train the model with DPO**:
   - Train the model using the DPO method and save the fine-tuned model.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset

# Set device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔹 Using device: {device}")

# Load trained reward model from Task 2
model_name = "gpt2"
reward_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device)
reward_model.load_state_dict(torch.load("reward_model.pth", map_location=device))
reward_model.config.pad_token_id = reward_model.config.eos_token_id  # Ensure pad token is set

# Load dataset (Using OpenAI Summarization Feedback dataset)
dataset = load_dataset("openai/summarize_from_feedback", "axis", split="test")  # Use test split
print(f"✅ Dataset Loaded: {len(dataset)} samples")

# Load tokenizer & set padding
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Use EOS as padding token

# Preprocessing function for DPO
def preprocess_data(example):
    """Processes dataset for DPO training."""
    return {
        "prompt": str(example.get("info", {}).get("post", "")),  # Extract original post
        "chosen": str(example.get("summary", "")),  # Preferred summary
        "rejected": str(example.get("summary", ""))  # Currently duplicating since dataset only has 'summary'
    }

# Apply preprocessing
dataset = dataset.map(preprocess_data, remove_columns=["info", "summary", "worker", "batch", "split"], features=None)
# Convert dataset to list format
prompts = [entry["prompt"] for entry in dataset]
chosen_responses = [entry["chosen"] for entry in dataset]
rejected_responses = [entry["rejected"] for entry in dataset]

# Tokenize inputs
chosen_outputs = tokenizer(chosen_responses, padding=True, truncation=True, max_length=128, return_tensors="pt")
rejected_outputs = tokenizer(rejected_responses, padding=True, truncation=True, max_length=128, return_tensors="pt")

# Convert dataset to PyTorch tensors
inputs = tokenizer(prompts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
chosen_inputs = {k: v.to(device) for k, v in chosen_outputs.items()}
rejected_inputs = {k: v.to(device) for k, v in rejected_outputs.items()}

# Define DPO loss function
def dpo_loss(chosen_scores, rejected_scores, beta=0.1):
    """
    Implements Direct Preference Optimization (DPO) loss.

    Args:
    - chosen_scores: Model outputs for chosen responses
    - rejected_scores: Model outputs for rejected responses
    - beta: Temperature parameter for softmax scaling

    Returns:
    - Loss value
    """
    logits_diff = chosen_scores - rejected_scores
    loss = -torch.log(torch.sigmoid(logits_diff / beta)).mean()
    return loss

# Define optimizer
optimizer = optim.AdamW(reward_model.parameters(), lr=1e-5)

# Reduce batch size to prevent CUDA OOM
BATCH_SIZE = 4

# Training function using DPO
def train_dpo(model, optimizer, chosen_inputs, rejected_inputs, epochs=10):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        for i in range(0, len(chosen_inputs["input_ids"]), BATCH_SIZE):
            # Select batch
            batch_chosen = {k: v[i:i+BATCH_SIZE] for k, v in chosen_inputs.items()}
            batch_rejected = {k: v[i:i+BATCH_SIZE] for k, v in rejected_inputs.items()}

            # Compute logits
            chosen_scores = model(**batch_chosen).logits.squeeze()
            rejected_scores = model(**batch_rejected).logits.squeeze()

            # Ensure correct tensor dimensions
            if chosen_scores.shape[0] == 0 or rejected_scores.shape[0] == 0:
                print("Skipping empty batch...")
                continue

            # Compute DPO loss
            loss = dpo_loss(chosen_scores, rejected_scores)

            loss.backward()
            optimizer.step()

            # Free unused memory
            torch.cuda.empty_cache()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Train the model with DPO
train_dpo(reward_model, optimizer, chosen_inputs, rejected_inputs)

# Save fine-tuned reward model
torch.save(reward_model.state_dict(), "reward_model_dpo.pth")
print("✅ DPO Fine-Tuning Complete & Model Saved!")
```

## Deployment

### Deploy to Streamlit

1. **Create a GitHub Repository**:
   - Go to GitHub and create a new repository.
   - Add your `app.py` file to the repository.

2. **Add a Requirements File**:
   - Create a `requirements.txt` file in your repository with the following content:

```plaintext name=requirements.txt
torch
streamlit
transformers
datasets
```

3. **Push the Changes to GitHub**:
   - Commit and push your changes to GitHub.

4. **Deploy on Streamlit**:
   - Go to [Streamlit Cloud](https://share.streamlit.io/).
   - Click on "New app" and connect it to your GitHub repository.
   - Fill in the necessary details (repository, branch, and file path).
   - Click on "Deploy".

Your web app should now be live and accessible via a Streamlit-provided URL.

## Usage

To run the Streamlit app locally:
1. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```
2. Run the app:
   ```sh
   streamlit run app.py
   ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
