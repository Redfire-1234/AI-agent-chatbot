import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

st.set_page_config(
    page_title="AI Agent Chatbot",
    page_icon="🤖",
    layout="wide"
)

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_REPO = "Redfire-1234/AI-agent"

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(base_model, LORA_REPO)
    model.eval()
    return tokenizer, model

def generate_response(tokenizer, model, user_input):
    """Generate response from the model"""
    inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean the response - remove the input if it's in the output
    if user_input in reply:
        reply = reply.replace(user_input, "").strip()
    
    return reply

def clear_chat():
    """Clear the chat history"""
    st.session_state.messages = []
    st.session_state.chat_input = ""

# Header with New Chat button
col1, col2 = st.columns([6, 1])
with col1:
    st.title("🤖 AI Agent Chatbot")
    st.caption("Powered by Qwen 2.5 + LoRA Fine-tuning")
with col2:
    if st.button("🗑️ New Chat", use_container_width=True, type="secondary"):
        clear_chat()
        st.rerun()

# Load model with status
with st.spinner("Loading model... (first time takes 2-3 minutes)"):
    tokenizer, model = load_model()

# Display chat history
chat_container = st.container()
with chat_container:
    if len(st.session_state.messages) == 0:
        st.info("👋 **Welcome to AI Agent Chatbot!**\n\n💬 Ask me anything and keep the conversation going.\n\n⚠️ **To end the conversation, simply type:** `q`")
    
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.write(message["content"])

# Chat input at the bottom
user_input = st.chat_input("Type your message here... (Type 'q' to end conversation)")

if user_input:
    # Check if user wants to quit
    if user_input.strip().lower() == 'q':
        st.session_state.messages.append({
            "role": "assistant",
            "content": "👋 Goodbye! Click 'New Chat' to start a fresh conversation."
        })
        st.rerun()
    
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Generate response
    with st.spinner("Thinking..."):
        reply = generate_response(tokenizer, model, user_input)
    
    # Add assistant response to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": reply
    })
    
    # Rerun to update the chat display
    st.rerun()

# Sidebar with settings and info
with st.sidebar:
    st.header("ℹ️ About")
    st.write(f"**Base Model:** Qwen 2.5 1.5B")
    st.write(f"**LoRA Adapter:** {LORA_REPO.split('/')[-1]}")
    
    st.divider()
    
    st.header("📊 Chat Stats")
    st.metric("Messages", len(st.session_state.messages))
    st.metric("User Messages", len([m for m in st.session_state.messages if m["role"] == "user"]))
    st.metric("Bot Messages", len([m for m in st.session_state.messages if m["role"] == "assistant"]))
    
    st.divider()
    
    st.header("💡 Tips")
    st.info("""
    - Type your question and press Enter
    - Type 'q' to end the conversation
    - Click 'New Chat' to start fresh
    - All messages are saved in this session
    """)
    
    st.divider()
    
    if st.button("🗑️ Clear History", use_container_width=True, type="primary"):
        clear_chat()
        st.rerun()
    
    st.divider()
    st.caption("Made with ❤️ using Streamlit")

