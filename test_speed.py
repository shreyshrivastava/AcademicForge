import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "google/gemma-2-2b-it"

def test_model():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Please set your HuggingFace token before running this script:")
        print("export HF_TOKEN='your_token_here'")
        return

    print(f"Testing {MODEL_NAME} via Transformers (mps)...")
    start_load = time.perf_counter()
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=device, token=hf_token)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
        
    load_time = time.perf_counter() - start_load
    print(f"Load time: {load_time:.2f}s")
    
    # Simulating the AcademicForge summarizer prompt
    prompt = """You are AcademicForge, a careful academic research assistant. Write concise plain-English paper summaries for builders in 2-4 sentences. DO NOT copy or echo the original abstract. You must synthesize and rewrite it in your own words. Avoid headings, bullet lists, hype, and conversational prefaces.

TITLE: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
ABSTRACT:
We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

---
TASK: Based on the abstract above, write a completely new 2-4 sentence summary in your own words. DO NOT copy the abstract."""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print("\nGenerating...")
    start_gen = time.perf_counter()
    outputs = model.generate(**inputs, max_new_tokens=200)
    
    # Extract only the newly generated text (ignoring the prompt)
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    gen_time = time.perf_counter() - start_gen
    
    print("\n--- Gemma-2-2B-it Response ---")
    print(response.strip())
    print("------------------------------")
    print(f"Generation time: {gen_time:.2f}s")

if __name__ == "__main__":
    test_model()
