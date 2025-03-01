import os
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

# Define the model name and directory
model_name = "gpt2"
model_dir = "reward_model"

# Create the directory if it doesn't exist
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Function to check if all required files exist
def check_files():
    required_files = [
        "config.json",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json"
    ]
    return all([os.path.exists(os.path.join(model_dir, file)) for file in required_files])

# Download files if they don't exist
if not check_files():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    print("Downloading model files...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print("Download complete.")
else:
    print("Model files already exist.")

# ✅ Set device (Use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load reward model (For scoring)
reward_model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=1).to(device)

# Load the state dict
state_dict = torch.load("reward_model.pth", map_location=device)
missing_keys, unexpected_keys = reward_model.load_state_dict(state_dict, strict=False)

# Check for missing or unexpected keys
if missing_keys:
    print(f"Missing keys: {missing_keys}")
if unexpected_keys:
    print(f"Unexpected keys: {unexpected_keys}")

reward_model.eval()

# ✅ Load text generation model (For generating responses)
response_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
response_tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ Load tokenizer and fix padding issue
reward_tokenizer = AutoTokenizer.from_pretrained(model_dir)
if reward_tokenizer.pad_token is None:
    reward_tokenizer.pad_token = reward_tokenizer.eos_token  # Set EOS token as padding

# 🎯 Function to generate AI response + Score
def generate_response(prompt):
    # Generate a response text using the language model
    input_ids = response_tokenizer.encode(prompt, return_tensors="pt").to(device)
    response_output = response_model.generate(input_ids, max_length=100, num_return_sequences=1)
    response_text = response_tokenizer.decode(response_output[0], skip_special_tokens=True)

    # Compute the reward score for the generated response
    inputs = reward_tokenizer(response_text, truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        output = reward_model(**inputs)

    # Convert logit score to probability
    score = output.logits.item()
    probability = torch.sigmoid(torch.tensor(score)).item()

    return response_text, probability  # Return response text + probability score

# 🌟 Modern & Professional UI with Streamlit
st.set_page_config(page_title="AI Response Evaluator", page_icon="💡", layout="wide")

# 🎨 Apply Custom CSS for a Professional Look
st.markdown("""
    <style>
        body {
            background-color: #f8f9fa;
            color: #212529;
        }
        .stApp {
            background: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #007bff;
            text-align: center;
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        .stTextInput>div>div>textarea {
            font-size: 1.1rem;
            padding: 10px;
            border-radius: 5px;
            background: #ffffff;
            color: #212529;
            border: 1px solid #007bff;
            box-shadow: inset 0px 0px 5px rgba(0, 0, 0, 0.1);
        }
        .stButton>button {
            background: #007bff;
            color: white;
            font-size: 1.1rem;
            border-radius: 5px;
            padding: 10px 20px;
            border: none;
            box-shadow: 0px 0px 10px rgba(0, 123, 255, 0.5);
        }
        .stButton>button:hover {
            background: #0056b3;
        }
        .stSuccess, .stInfo {
            font-size: 1.1rem;
            background: #ffffff;
            color: #212529;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #007bff;
            box-shadow: 0px 0px 10px rgba(0, 123, 255, 0.1);
        }
        .footer {
            text-align: center;
            font-size: 1rem;
            color: #6c757d;
            padding-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# 🎨 Professional Header
st.markdown("<h1>💡 AI Response Evaluator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:1.3rem; color:#007bff;'>Input a prompt to see how the AI evaluates different responses!</p>", unsafe_allow_html=True)

# 📌 User Input with large Text Box
user_input = st.text_area("Enter your prompt:", "What is the capital of France?", height=150)

# 🔍 Evaluate Button with Icon
if st.button("🔍 Evaluate Response"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a valid prompt!")
    else:
        ai_response, probability = generate_response(user_input)
        
        # 🎯 Display the result with professional styling
        st.markdown("<h2 style='text-align:center; color:#007bff;'>📊 AI Evaluation:</h2>", unsafe_allow_html=True)
        st.success(f"**AI Response:** {ai_response}")  # Show AI-generated response
        st.info(f"**AI Confidence Score: {probability:.4f}** (Higher is better)")  # Show confidence score
        
        # 🎨 Interpret results in a more engaging way
        if probability > 0.8:
            st.markdown("<p style='text-align:center; font-size:1.2rem; color:#155724; background:#d4edda; padding:10px; border-radius:5px;'>✅ The AI finds this response highly relevant!</p>", unsafe_allow_html=True)
        elif probability > 0.5:
            st.markdown("<p style='text-align:center; font-size:1.2rem; color:#856404; background:#fff3cd; padding:10px; border-radius:5px;'>⚖️ The AI finds this response moderately relevant.</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='text-align:center; font-size:1.2rem; color:#721c24; background:#f8d7da; padding:10px; border-radius:5px;'>❌ The AI finds this response less relevant.</p>", unsafe_allow_html=True)

# 🌟 Footer with Your Name & ID
st.markdown("<p class='footer'>Built using Streamlit & PyTorch. <br> <strong>Created by St125050</strong></p>", unsafe_allow_html=True)
