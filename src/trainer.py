# trainer.py

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import config
import data_loader
import model
import os
import torch

class CustomDataCollator:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
    
    def __call__(self, features):
        # Extract input_ids, attention_mask, and labels
        input_ids = [f['input_ids'] for f in features]
        attention_mask = [f['attention_mask'] for f in features]
        labels = [f['labels'] for f in features]
        
        # Find max length for input and labels separately
        max_input_len = max(len(ids) for ids in input_ids)
        max_label_len = max(len(label) for label in labels)
        max_len = max(max_input_len, max_label_len)
        
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        
        for i in range(len(features)):
            # Pad input_ids
            input_pad_len = max_len - len(input_ids[i])
            padded_input_ids.append(input_ids[i] + [self.tokenizer.pad_token_id] * input_pad_len)
            padded_attention_mask.append(attention_mask[i] + [0] * input_pad_len)
            
            # Pad labels
            label_pad_len = max_len - len(labels[i])
            padded_labels.append(labels[i] + [-100] * label_pad_len)
        
        return {
            'input_ids': torch.tensor(padded_input_ids),
            'attention_mask': torch.tensor(padded_attention_mask),
            'labels': torch.tensor(padded_labels)
        }

def tokenize_function(examples, tokenizer):
    """Tokenize input and output texts."""
    # Handle both single examples and batches
    if isinstance(examples['input_text'], str):
        # Single example
        model_inputs = tokenizer(examples['input_text'], max_length=config.MAX_INPUT_LENGTH, truncation=True)
        labels = tokenizer(examples['output_text'], max_length=config.MAX_OUTPUT_LENGTH, truncation=True)
        model_inputs['labels'] = labels['input_ids']
    else:
        # Batch of examples
        model_inputs = tokenizer(examples['input_text'], max_length=config.MAX_INPUT_LENGTH, truncation=True, padding=False)
        labels = tokenizer(examples['output_text'], max_length=config.MAX_OUTPUT_LENGTH, truncation=True, padding=False)
        model_inputs['labels'] = labels['input_ids']
    return model_inputs

def main():
    # Load data
    dataset = data_loader.load_and_preprocess()

    # Load tokenizer and model
    tokenizer = model.get_tokenizer()
    ml_model = model.get_model()

    # Tokenize dataset
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # Data collator
    data_collator = CustomDataCollator(tokenizer=tokenizer, model=ml_model)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        save_steps=500,
        save_strategy="epoch",
        logging_steps=100,
        fp16=True,  # Mixed precision for speed
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=ml_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    print(f"Starting training with output_dir: {config.OUTPUT_DIR}")
    trainer.train()
    print("Training completed. Saving final model.")
    trainer.save_model(config.OUTPUT_DIR)
    save_path = os.path.abspath(config.OUTPUT_DIR)
    print(f"Model saved to: {save_path}")

if __name__ == "__main__":
    main()