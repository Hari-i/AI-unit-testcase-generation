# generator.py: Inference script to generate test cases for user-provided functions

import os
from transformers import pipeline
import config
import model
import utils

def generate_tests(function_code):
    """
    Generate unit tests for a given function code string.
    Returns a string of assert statements.
    """
    # Verify trained model directory exists
    model_path = config.OUTPUT_DIR
    if not os.path.exists(model_path) or not os.path.isdir(model_path):
        raise FileNotFoundError(f"Trained model directory '{model_path}' not found. Run trainer.py first.")
    if not any(fname.endswith(('.bin', '.safetensors')) for fname in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, fname))):
        raise FileNotFoundError(f"Model weights not found in '{model_path}'. Ensure training completed successfully.")

    print(f"Loading model from: {os.path.abspath(model_path)}")

    # Load tokenizer and model
    tokenizer = model.get_tokenizer()
    ml_model = model.get_model().from_pretrained(model_path)

    # Prepare input text with a guiding instruction
    input_text = config.PROMPT_PREFIX + function_code + "\nUse scalar inputs for tests (e.g., multiply(2, 3))."

    # Generate using pipeline
    generator = pipeline('text2text-generation', model=ml_model, tokenizer=tokenizer)
    generated = generator(input_text, max_length=config.MAX_OUTPUT_LENGTH, num_beams=config.NUM_BEAMS, no_repeat_ngram_size=2)[0]['generated_text']

    # Post-process and return clean test cases
    return utils.postprocess_generated_text(generated)

def main():
    print("Enter your Python function code (end with EOF or Ctrl+Z):")
    function_code = ""
    while True:
        try:
            line = input()
            function_code += line + "\n"
        except EOFError:
            break
        except KeyboardInterrupt:
            break

    if function_code.strip():
        tests = generate_tests(function_code.strip())
        print("\nGenerated Unit Tests:\n")
        print(tests)
    else:
        print("No function provided.")

if __name__ == "__main__":
    main()