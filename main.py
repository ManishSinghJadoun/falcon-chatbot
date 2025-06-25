from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

app = FastAPI()

# Load TinyLlama model and tokenizer (once at startup)
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Input schema
class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return {"message": "TinyLlama Chatbot is running on Render!"}

@app.post("/chat")
def chat(request: ChatRequest):
    prompt = f"<|system|>\nYou are a helpful assistant.\n<|user|>\n{request.message}\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    reply = response.split("<|assistant|>")[-1].strip()
    return {"response": reply}
