from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"      # official base model
LORA_REPO = "Redfire-1234/AI-agent"            # your repo containing LoRA files

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load LoRA on top of base model
    model = PeftModel.from_pretrained(
        base_model,
        LORA_REPO
    )

    return tokenizer, model

