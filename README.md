AI Chatbot Fine-Tuning using Qwen2.5 + LoRA + Flask (Dockerized Deployment)  

This project fine-tunes the Qwen/Qwen2.5-1.5B-Instruct large language model using LoRA (Low-Rank Adaptation) on a custom dataset created from multiple social media CSV files.  
The trained model is exported, tested, pushed to Hugging Face, and finally deployed using a Dockerized Flask API.  

The entire pipeline includes:  
✔ Dataset creation  
✔ Cleaning & preprocessing  
✔ Prompt-response generation  
✔ LoRA-based fine-tuning  
✔ Model evaluation  
✔ Pushing model to Hugging Face Hub  
✔ Chat inference (single & batch)  
✔ Docker + Flask deployment  

References:  
Qwen2.5 Model: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct  
PEFT LoRA Library: https://github.com/huggingface/peft  
Transformers Library: https://github.com/huggingface/transformers  
Dataset Source: Custom CSVs uploaded to HuggingFace Dataset Repo  

Project Workflow  
1] Setup  
Install required libraries (transformers, peft, accelerate, bitsandbytes, etc.)  
Login to HuggingFace  
Download and unzip the dataset from your HF repo  
Verify dataset structure  

2] Prepare Dataset  
Steps performed:  
Load all CSV files  
Add a platform column based on file name  
Remove duplicates  
Drop null rows  
Clean text (URLs, non-ASCII, tags)  
Keep rows with text length > 5  
Final dataset saved as:  
        ai_agent_cleaned.csv   

3] Create Chat-style Instruction Dataset  
Each row is converted into an input/output pair:  
Input examples:  
Summarize this twitter post: …  
What is this reddit post about? …  
Analyze this youtube comment: …  
Output examples:  
Sentiment classification (positive / negative / neutral)  
A short summary  
Description of tone  
Final dataset saved as:  
       chatbot_dataset.csv  

4] Tokenization  
Tokenize using Qwen tokenizer  
Format each sample as:  
       input + "\n" + output   
Create train/test split  
Limit training set to ~3000 samples for speed  
Label tokens = input_ids (standard supervised fine-tuning)  

5] Load Qwen Model with QLoRA  
Load Qwen2.5-1.5B-Instruct in 4-bit NF4 quantization  
Enable gradient checkpointing  
Apply LoRA on key transformer modules:  
      q_proj, k_proj, v_proj, o_proj  
      gate_proj, up_proj, down_proj  

6] Train the Model  
Training configuration:  
      batch_size=1  
      gradient_accumulation_steps=2  
      num_epochs=1  
      lr=2e-4  
      fp16=True  
Training performed using HuggingFace Trainer API.  
Final LoRA weights + tokenizer saved to:  
      my-chatbot-model/  
The folder is zipped and exported for download.  

7] Load & Test the Fine-tuned Chatbot  
A custom inference function was implemented:  
Cleans user input  
Builds chat prompt  
Generates text using:  
      top-p sampling  
      temperature  
      repetition penalty  
      no-repeat n-gram  
Example test queries included:  
      Summaries  
      Chat-like questions  
      General knowledge  
The model responds smoothly to chat instructions.  

8] Push Model to Hugging Face Hub  
Using:  
      model.push_to_hub("AI-agent")  
      tokenizer.push_to_hub("AI-agent")  
Final model URL:  
https://huggingface.co/spaces/Redfire-1234/AI-chatbot  

9] Batch Inference  
A helper function allows generating multiple responses at once:  
      batch_generate(["text1","text2","text3"])  
Useful for:  
testing  
dataset expansion  
evaluation  

10] Save Conversation History  
Utility function saves chat logs to text file:  
      conversation.txt  

11] Technologies Used  
Python     
Google Colab  
Hugging Face Hub  
Qwen2.5  
LoRA / PEFT  
Transformers  
BitsAndBytes (4-bit quantization)  
Pandas  


