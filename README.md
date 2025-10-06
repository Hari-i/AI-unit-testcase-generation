# AI Unit Test Case Generator

An intelligent system that automatically generates unit test cases for Python functions using CodeT5 and the MBPP dataset.

## Overview

This project uses a fine-tuned CodeT5 model to learn the mapping from Python function code to corresponding unit test cases. The model is trained on the MBPP (Mostly Basic Python Problems) dataset, which contains Python functions along with their test cases.

## Features

- **CodeT5-based**: Uses Salesforce's CodeT5-small model for code-to-text generation
- **MBPP Dataset**: Trained on comprehensive Python function examples with AST validation
- **Automatic Generation**: Generates unit tests for user-provided Python functions
- **Robust Processing**: Includes syntax validation and function metadata extraction
- **Command-Line Interface**: Simple CLI for test generation

## Quick Start

1. **Train the model** (if not already trained):

```bash
python src/trainer.py
```

2. **Generate tests**:

```bash
python main.py
```

## Usage

### Training the Model

To train the model on the MBPP dataset:

```bash
cd src
python trainer.py
```

This will:

- Load and preprocess the MBPP dataset
- Fine-tune the CodeT5 model
- Save the trained model to `./trained_model/`

### Generating Test Cases

After training, you can generate test cases for your functions:

```bash
cd src
python generator.py
```

Then input your Python function code and press Ctrl+Z (Windows) or Ctrl+D (Linux/Mac) to generate test cases.

### Example Usage

```python
# Input function
def add_numbers(a, b):
    return a + b

# Generated test cases
assert add_numbers(2, 3) == 5
assert add_numbers(0, 0) == 0
assert add_numbers(-1, 1) == 0
```

## Project Structure

```
AI unit testcase generation/
├── main.py                # Main entry script
├── requirements.txt       # Dependencies
├── README.md              # Documentation
├── src/                   # Core source code
│   ├── config.py          # Configuration
│   ├── data_loader.py     # Data processing
│   ├── model.py           # Model setup
│   ├── trainer.py         # Training pipeline
│   ├── generator.py       # Test generation
│   └── utils.py           # Utilities
├── data/                  # Dataset
│   └── mbpp.jsonl         # MBPP dataset
└── trained_model/         # Trained model
    ├── model.safetensors  # Model weights
    ├── config.json        # Model configuration
    └── ...                # Other model files
```

## Configuration

Key settings in `src/config.py`:

- `MODEL_NAME`: CodeT5 model variant (default: "Salesforce/codet5-small")
- `NUM_EPOCHS`: Training epochs (default: 10)
- `BATCH_SIZE`: Training batch size (default: 4)
- `LEARNING_RATE`: Learning rate (default: 5e-5)
- `MAX_INPUT_LENGTH`: Maximum input sequence length (default: 512)
- `MAX_OUTPUT_LENGTH`: Maximum output sequence length (default: 256)

## Dataset

The MBPP dataset contains:

- Python function implementations
- Corresponding unit test cases
- Function descriptions and metadata
- AST-validated code structure

## Model Architecture

- **Base Model**: Salesforce CodeT5-small
- **Task**: Text-to-text generation (function code → test cases)
- **Training**: Fine-tuning with Seq2SeqTrainer
- **Inference**: Beam search with configurable parameters

## Performance Notes

- Training time depends on hardware (GPU recommended)
- Model size: ~220MB (CodeT5-small)
- Memory requirements: ~4GB RAM minimum
- Training on full MBPP dataset: ~30-60 minutes on modern GPU

## Troubleshooting

1. **CUDA/GPU Issues**: Ensure PyTorch is installed with CUDA support
2. **Memory Issues**: Reduce batch size in config.py
3. **Model Loading**: Ensure training completed successfully before inference
4. **Dataset Issues**: Verify mbpp.jsonl exists in data/ directory

## Future Improvements

- Support for more complex test case generation
- Integration with popular testing frameworks (pytest, unittest)
- Web interface for easier interaction
- Support for multiple programming languages
- Enhanced prompt engineering for better test quality

## License

This project is for educational and research purposes. Please respect the licenses of the underlying models and datasets.
