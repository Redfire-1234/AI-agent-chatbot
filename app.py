import streamlit as st

# Page config first
st.set_page_config(
    page_title="AI Agent Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Import with error handling
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.info("Please check the error logs in 'Manage App' → Terminal")
    st.stop()

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_REPO = "Redfire-1234/AI-agent"

@st.cache_resource(show_spinner=False)
def load_model():
    """Load model with progress tracking"""
    try:
        with st.spinner("🔄 Loading tokenizer..."):
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        
        with st.spinner("🔄 Loading base model... (this may take 2-3 minutes)"):
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float32,  # Use float32 for CPU
                low_cpu_mem_usage=True
            )
        
        with st.spinner("🔄 Loading LoRA adapter..."):
            model = PeftModel.from_pretrained(base_model, LORA_REPO)
            model.eval()
        
        return tokenizer, model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("This might be a memory issue. Try restarting the app.")
        st.stop()

def generate_response(tokenizer, model, user_input, max_tokens=200, temperature=0.7):
    """Generate response"""
    try:
        inputs = tokenizer(user_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input from output
        if user_input in reply:
            reply = reply.replace(user_input, "").strip()
        
        return reply
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

# ------------------------------- UI ------------------------------

st.title("🤖 AI Agent Chatbot")
st.caption("Powered by Qwen 2.5 + LoRA Fine-tuning")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    max_tokens = st.slider("Max Tokens", 50, 300, 200)
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.1)
    
    st.divider()
    st.info("💡 **Tip:** Higher temperature = more creative responses")

# Load model
with st.status("Loading model...", expanded=True) as status:
    try:
        tokenizer, model = load_model()
        status.update(label="✅ Model loaded!", state="complete")
    except Exception as e:
        status.update(label="❌ Failed to load model", state="error")
        st.stop()

# Input
user_input = st.text_area(
    "Your Question:",
    placeholder="Ask me anything...",
    height=100,
    key="input"
)

# Buttons
col1, col2 = st.columns([1, 5])
with col1:
    generate = st.button("🚀 Send", use_container_width=True)

if generate:
    if not user_input.strip():
        st.warning("⚠️ Please enter a question")
    else:
        with st.spinner("Thinking..."):
            reply = generate_response(
                tokenizer,
                model,
                user_input,
                max_tokens=max_tokens,
                temperature=temperature
            )
        
        st.divider()
        st.subheader("💬 Response:")
        st.write(reply)

