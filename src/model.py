# model.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import config

def get_tokenizer():
    """Return the tokenizer for the model."""
    print(f"TOKENIZER_NAME: {config.TOKENIZER_NAME}, type: {type(config.TOKENIZER_NAME)}")
    return RobertaTokenizer.from_pretrained(config.TOKENIZER_NAME)

def get_model():
    """Return the pre-trained model for fine-tuning."""
    return T5ForConditionalGeneration.from_pretrained(config.MODEL_NAME)