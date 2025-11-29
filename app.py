import streamlit as st
import torch
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="AI Agent Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Try importing with error handling
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    logger.info("✓ Successfully imported transformers and peft")
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.info("Installing required packages... Please wait and refresh.")
    st.stop()

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_REPO = "Redfire-1234/AI-agent"

@st.cache_resource
def load_model():
    """Load the model and tokenizer with error handling"""
    try:
        with st.spinner("Loading model... This may take a few minutes on first run."):
            logger.info(f"Loading tokenizer from {BASE_MODEL}")
            tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL,
                trust_remote_code=True
            )
            
            logger.info(f"Loading base model from {BASE_MODEL}")
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            logger.info(f"Loading LoRA adapter from {LORA_REPO}")
            model = PeftModel.from_pretrained(
                base_model,
                LORA_REPO
            )
            model.eval()
            
            logger.info("✓ Model loaded successfully")
            return tokenizer, model
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

def generate_response(tokenizer, model, user_input, max_tokens=300, temperature=0.7):
    """Generate response with better formatting"""
    try:
        # Format input
        messages = [{"role": "user", "content": user_input}]
        
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            prompt = f"User: {user_input}\nAssistant:"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode only new tokens
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        reply = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return reply.strip()
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"Error: {str(e)}"

# ------------------------------- UI ------------------------------

st.title("🤖 AI Agent - LoRA Fine-Tuned Model")
st.write("Ask anything below 👇")

# Add sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.write(f"**Base Model:** {BASE_MODEL}")
    st.write(f"**LoRA Adapter:** {LORA_REPO}")
    st.divider()
    
    st.header("⚙️ Settings")
    max_tokens = st.slider("Max Tokens", 50, 500, 300, 50)
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.1)
    
    st.divider()
    st.caption("Made with ❤️ using Streamlit")

# Load model
try:
    tokenizer, model = load_model()
    st.success("✓ Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

# Input
user_input = st.text_area(
    "Your Question",
    placeholder="Type your question here...",
    height=100
)

col1, col2 = st.columns([1, 5])
with col1:
    generate_btn = st.button("🚀 Generate", use_container_width=True)
with col2:
    clear_btn = st.button("🗑️ Clear", use_container_width=True)

if clear_btn:
    st.rerun()

if generate_btn:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a question")
    else:
        with st.spinner("Generating response..."):
            reply = generate_response(
                tokenizer, 
                model, 
                user_input,
                max_tokens=max_tokens,
                temperature=temperature
            )
        
        st.divider()
        st.subheader("📝 Response:")
        st.write(reply)

# Add conversation history
if "history" not in st.session_state:
    st.session_state.history = []

if generate_btn and user_input.strip():
    st.session_state.history.append({
        "question": user_input,
        "answer": reply
    })

# Display history
if st.session_state.history:
    st.divider()
    st.subheader("💬 Conversation History")
    
    for i, chat in enumerate(reversed(st.session_state.history[-5:])):
        with st.expander(f"Chat {len(st.session_state.history) - i}"):
            st.write("**Q:**", chat["question"])
            st.write("**A:**", chat["answer"])


