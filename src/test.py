# test_tokenizer.py (place in src)
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from transformers import T5Tokenizer
import config
print(f"TOKENIZER_NAME: {config.TOKENIZER_NAME}, type: {type(config.TOKENIZER_NAME)}")
tokenizer = T5Tokenizer.from_pretrained(config.TOKENIZER_NAME, use_fast=False)
print(f"Tokenizer type: {type(tokenizer).__name__}")
print("Tokenizer loaded successfully!")