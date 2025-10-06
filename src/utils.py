# utils.py: Utility functions

def postprocess_generated_text(generated_text):
    """Clean up generated text to format unit tests."""
    # Split by lines and filter for assert statements
    lines = []
    for line in generated_text.split('\n'):
        line = line.strip()
        if line and "assert" in line.lower():
            # Clean up the line - remove extra spaces and fix formatting
            line = ' '.join(line.split())  # Remove extra whitespace
            lines.append(line)
    
    # If no assert statements found, return a message
    if not lines:
        return "No valid test cases generated. The model may need more training."
    
    return "\n".join(lines)