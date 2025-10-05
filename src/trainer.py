# trainer.py

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, RobertaTokenizer
import config
import data_loader
import model
import os

def tokenize_function(example, tokenizer):
    """Tokenize input and output texts."""
    model_inputs = tokenizer(example['input_text'], max_length=config.MAX_INPUT_LENGTH, truncation=True)
    labels = tokenizer(example['output_text'], max_length=config.MAX_OUTPUT_LENGTH, truncation=True)
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
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=ml_model)

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