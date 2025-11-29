import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_REPO = "Redfire-1234/AI-agent"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(
        base_model,
        LORA_REPO
    )

    return tokenizer, model

# ------------------------------- UI ------------------------------

st.title("🤖 AI Agent - LoRA Fine-Tuned Model")

st.write("Ask anything below 👇")

tokenizer, model = load_model()

user_input = st.text_input("Your Question")

if st.button("Generate Response"):
    if user_input.strip() == "":
        st.warning("Please enter a question")
    else:
        inputs = tokenizer(user_input, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True
            )

        reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.success(reply)


