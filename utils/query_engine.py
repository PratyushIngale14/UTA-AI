from utils.embed import load_vectorstore, model
from utils.prompts import quiz_prompt_template, notes_prompt_template
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
phi_model = AutoModelForCausalLM.from_pretrained("distilgpt2")

def retrieve_top_k(k=3):
    index, chunks = load_vectorstore()
    last_query = "educational content"
    query_vec = model.encode([last_query])
    D, I = index.search(query_vec, k)
    return [chunks[i] for i in I[0]]

def generate_notes():
    context = "\n".join(retrieve_top_k())
    prompt = notes_prompt_template.format(context=context)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    outputs = phi_model.generate(**inputs, max_new_tokens=300)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_quiz():
    context = "\n".join(retrieve_top_k())
    prompt = quiz_prompt_template.format(context=context)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    outputs = phi_model.generate(**inputs, max_new_tokens=300)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
