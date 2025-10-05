# model.py: Load tokenizer and model

from transformers import T5Tokenizer, T5ForConditionalGeneration
import config

def get_tokenizer():
    """Return the tokenizer for the model."""
    return T5Tokenizer.from_pretrained('config.TOKENIZER_NAME')

def get_model():
    """Return the pre-trained model for fine-tuning."""
    return T5ForConditionalGeneration.from_pretrained(config.MODEL_NAME)