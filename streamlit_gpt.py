import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load GPT-2
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

def generate_text(prompt, max_length=150, temperature=0.7, top_p=0.9):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = torch.ones_like(inputs)
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=0,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit UI
st.title("🛠 Tool Description Generator (GPT-2)")
st.markdown("Generate coherent paragraphs about software tools using GPT-2.")

user_prompt = st.text_area("Enter your prompt:", "Explain how Docker helps developers.", height=150)

col1, col2 = st.columns([1, 2])
with col1:
    max_len = st.slider("Max length", 50, 300, 150)
with col2:
    temp = st.slider("Temperature", 0.1, 1.5, 0.7)

if st.button("Generate"):
    with st.spinner("Generating..."):
        output = generate_text(user_prompt, max_length=max_len, temperature=temp)
    st.subheader("Generated Text:")
    st.write(output)
