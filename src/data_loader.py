# data_loader.py: Load and preprocess the locally saved MBPP dataset from JSONL

import json
import ast
import config
import os
from datasets import Dataset

def preprocess_mbpp(example):
    """Preprocess MBPP example: Combine code and tests into input-output pairs with AST validation."""
    code = example['code'].strip()
    if not code:
        return None

    # Validate code syntax and structure with AST
    try:
        tree = ast.parse(code)
        # Ensure the code defines at least one function
        func_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        if not func_nodes:
            return None  # Skip if no function definition found
        # Extract function name and parameters from the first function (assuming single function per example)
        func_node = func_nodes[0]
        func_name = func_node.name
        param_names = [arg.arg for arg in func_node.args.args]
        # Enrich input_text with metadata for better prompt guidance
        metadata = f"# Function: {func_name}, Parameters: {', '.join(param_names)}"
    except SyntaxError:
        return None  # Skip invalid syntax

    input_text = config.PROMPT_PREFIX + code + "\n" + metadata
    output_text = "\n".join(example['test_list']) if 'test_list' in example and example['test_list'] else ""
    return {'input_text': input_text, 'output_text': output_text}

def load_and_preprocess():
    """Load the locally saved MBPP dataset from JSONL, preprocess into input-output pairs."""
    # Use __file__ to get the script's directory and navigate to the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Move up from src to root
    data_path = os.path.join(project_root, "data", "mbpp.jsonl")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Local dataset not found at {data_path}. Download mbpp.jsonl from GitHub to {project_root}/data/.")

    # Load JSONL dataset
    print(f"Loading dataset from {data_path}...")
    examples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                example = json.loads(line.strip())
                processed = preprocess_mbpp(example)
                if processed is not None:
                    examples.append(processed)
            except json.JSONDecodeError:
                continue  # Skip invalid lines

    # Create a simple dataset from the list
    preprocessed_dataset = Dataset.from_list(examples)
    print(f"Loaded {len(preprocessed_dataset)} examples after preprocessing.")
    return preprocessed_dataset