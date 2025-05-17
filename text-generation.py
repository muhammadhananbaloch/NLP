import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Create Tokenizer and Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Story line's opening
seed_text = "Ibtesam is a rapper from Pakistan"

# Create input and outputs
input_ids = tokenizer.encode(seed_text, return_tensors='pt')
output = model.generate(input_ids, max_length=150, temperature=0.7, no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id)

# Generate text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
