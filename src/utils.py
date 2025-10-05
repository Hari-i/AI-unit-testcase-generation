# utils.py: Utility functions

def postprocess_generated_text(generated_text):
    """Clean up generated text to format unit tests."""
    lines = [line.strip() for line in generated_text.split('\n') if line.strip() and "assert" in line]
    return "\n".join(lines)