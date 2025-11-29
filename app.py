import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Your HuggingFace model link
HF_MODEL = "Redfire-1234/AI-agent"

st.title("🤖 My Custom ChatGPT")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, model

tokenizer, model = load_model()

user_input = st.text_input("Enter your message:")

if st.button("Generate Response"):
    inputs = tokenizer(user_input, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    st.subheader("Response:")
    st.write(response)
