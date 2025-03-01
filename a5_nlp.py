import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

# ✅ Set device (Use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load reward model (For scoring)
reward_model_name = "gpt2"
reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name, num_labels=1).to(device)
reward_model.load_state_dict(torch.load("reward_model.pth", map_location=device))
reward_model.eval()

# ✅ Load text generation model (For generating responses)
response_model_name = "gpt2"  # You can change this to another fine-tuned model
response_model = AutoModelForCausalLM.from_pretrained(response_model_name).to(device)
response_tokenizer = AutoTokenizer.from_pretrained(response_model_name)

# ✅ Load tokenizer and fix padding issue
reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
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

# 🌟 Modern & Colorful UI with Streamlit
st.set_page_config(page_title="AI Response Evaluator", page_icon="🤖", layout="wide")

# 🎨 Apply Custom CSS for Stylish Look
st.markdown("""
    <style>
        body {
            background-color: #282c34;
            color: #61dafb;
        }
        .stApp {
            background: linear-gradient(145deg, #282c34, #61dafb);
            padding: 3rem;
            border-radius: 15px;
        }
        h1 {
            color: #61dafb;
            text-align: center;
            font-size: 3rem;
        }
        .stTextInput>div>div>input {
            font-size: 1.2rem;
            padding: 10px;
            border-radius: 10px;
        }
        .stButton>button {
            background: #61dafb;
            color: #282c34;
            font-size: 1.2rem;
            border-radius: 10px;
            padding: 12px 24px;
        }
        .stButton>button:hover {
            background: #21a1f1;
        }
        .stSuccess {
            font-size: 1.2rem;
            background: #198754;
            color: #fff;
            padding: 15px;
            border-radius: 10px;
        }
        .footer {
            text-align: center;
            font-size: 1.1rem;
            color: #61dafb;
            padding-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# 🎨 Stylish Header
st.markdown("<h1>🤖 AI Response Evaluator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:1.3rem;'>Enter a text prompt and see how the AI rates different responses!</p>", unsafe_allow_html=True)

# 📌 User Input with large Text Box
user_input = st.text_area("Enter your prompt:", "What is the capital of France?", height=150)

# 🔍 Evaluate Button with Icon
if st.button("🔍 Evaluate Response"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a valid prompt!")
    else:
        ai_response, probability = generate_response(user_input)
        
        # 🎯 Display the result with modern styling
        st.markdown("<h2 style='text-align:center;'>📝 AI Evaluation:</h2>", unsafe_allow_html=True)
        st.success(f"**AI Response:** {ai_response}")  # Show AI-generated response
        st.info(f"**AI Confidence Score: {probability:.4f}** (Higher is better)")  # Show confidence score
        
        # 🎨 Interpret results in a more engaging way
        if probability > 0.8:
            st.markdown("<p style='text-align:center; font-size:1.2rem; background:#198754; color:#fff; padding:10px; border-radius:10px;'>✅ AI finds this response highly relevant!</p>", unsafe_allow_html=True)
        elif probability > 0.5:
            st.markdown("<p style='text-align:center; font-size:1.2rem; background:#ffc107; color:#000; padding:10px; border-radius:10px;'>⚖️ AI finds this response neutral.</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='text-align:center; font-size:1.2rem; background:#dc3545; color:#fff; padding:10px; border-radius:10px;'>❌ AI finds this response less relevant.</p>", unsafe_allow_html=True)

# 🌟 Footer with Your Name & ID
st.markdown("<p class='footer'>Built with ❤️ using Streamlit & PyTorch. <br> <strong>Created by Ponkrit ST124960</strong></p>", unsafe_allow_html=True)
