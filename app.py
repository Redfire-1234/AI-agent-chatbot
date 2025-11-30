import streamlit as st
import torch
import gc

st.set_page_config(
    page_title="AI Agent Chatbot",
    page_icon="🤖",
    layout="centered"
)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.stop()

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_REPO = "Redfire-1234/AI-agent"

@st.cache_resource(show_spinner=False)
def load_model():
    """Load model with aggressive memory optimization"""
    try:
        # Clear any cached memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        st.info("🔄 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        
        st.info("🔄 Loading model with 8-bit quantization... (this reduces memory by 75%)")
        
        # 8-bit quantization config
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            max_memory={0: "900MB", "cpu": "2GB"}  # Limit memory usage
        )
        
        st.info("🔄 Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, LORA_REPO)
        model.eval()
        
        return tokenizer, model
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.error("⚠️ **Memory Error:** Streamlit Cloud doesn't have enough RAM for this model.")
        st.info("👉 **Solution:** Deploy to Hugging Face Spaces (16GB RAM free) instead!")
        st.stop()

def generate_response(tokenizer, model, user_input, max_tokens=150, temperature=0.7):
    """Generate response with memory cleanup"""
    try:
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
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
        
        # Cleanup
        del inputs, outputs
        gc.collect()
        
        return reply.strip()
        
    except Exception as e:
        return f"Error: {str(e)}"

# ------------------------------- UI ------------------------------

st.title("🤖 AI Agent Chatbot")
st.caption("⚠️ Running on limited memory - may be slow")

# Warning banner
st.warning("⚠️ **Note:** This app requires ~3GB RAM but Streamlit Cloud only provides 1GB. If crashes occur, please use the Hugging Face Spaces deployment instead.")

with st.sidebar:
    st.header("⚙️ Settings")
    max_tokens = st.slider("Max Tokens", 50, 150, 100)  # Reduced max
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.1)
    
    st.divider()
    st.error("⚠️ Memory Limited Mode")
    st.caption("Reduced token limit to save memory")

# Load model
try:
    with st.status("Loading model...", expanded=True) as status:
        tokenizer, model = load_model()
        status.update(label="✅ Model loaded!", state="complete")
        st.success("✅ Ready! Ask me anything below.")
except:
    st.stop()

# Input
user_input = st.text_area("Your Question:", placeholder="Ask me anything...", height=100)

if st.button("🚀 Send", use_container_width=True):
    if not user_input.strip():
        st.warning("⚠️ Please enter a question")
    else:
        with st.spinner("Thinking..."):
            reply = generate_response(tokenizer, model, user_input, max_tokens=max_tokens, temperature=temperature)
        
        st.divider()
        st.subheader("💬 Response:")
        st.write(reply)

