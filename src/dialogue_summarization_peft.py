import mlflow
import argparse
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
import torch
import evaluate
from peft import LoraConfig, get_peft_model, TaskType

def tokenize_and_prepare_data(dataset_name, model_name):
    """
    Tokenizes the dataset and prepares it for training and evaluation.
    """
    dataset = load_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(example):
        start_prompt = 'Summarize the following conversation.\n\n'
        end_prompt = '\n\nSummary: '
        prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
        example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
        example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids
        return example

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary'])
    tokenized_datasets = tokenized_datasets.filter(lambda example, index: index % 100 == 0, with_indices=True)

    return tokenized_datasets

def fine_tune_with_peft(tokenized_datasets, model_name, output_dir, num_train_epochs, train_batch_size, eval_batch_size):
    """
    Fine-tunes the model using PEFT on the tokenized dataset.
    """
    original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    peft_model = get_peft_model(original_model, lora_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-3,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"]
    )

    trainer.train()

def evaluate_model(dataset_name, model_name, tokenizer, num_samples=10):
    """
    Evaluates the model using a subset of the dataset and prints ROUGE scores.
    """
    dataset = load_dataset(dataset_name)
    dialogues = dataset['test'][:num_samples]['dialogue']
    human_baseline_summaries = dataset['test'][:num_samples]['summary']
    model_summaries = []

    for dialogue in dialogues:
        prompt = f"Summarize the following conversation.\n\n{dialogue}\n\nSummary: "
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        model_output = peft_model.generate(input_ids=input_ids, max_new_tokens=200)
        model_summary = tokenizer.decode(model_output[0], skip_special_tokens=True)
        model_summaries.append(model_summary)

    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=model_summaries, references=human_baseline_summaries, use_stemmer=True)

    print('MODEL ROUGE SCORES:')
    print(results)

def main():
    # enable autologging
    mlflow.autolog()

    parser = argparse.ArgumentParser(description="Fine-tune and evaluate a dialogue summarization model with PEFT")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name to use for training and evaluation")
    parser.add_argument("--model_name", type=str, default='google/flan-t5-base', help="Model name or path")
    parser.add_argument("--output_dir", type=str, default='./peft_model', help="Output directory for saving the model")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to use for evaluation")

    args = parser.parse_args()

    tokenized_datasets = tokenize_and_prepare_data(args.dataset_name, args.model_name)
    fine_tune_with_peft(tokenized_datasets, args.model_name, args.output_dir, args.num_train_epochs, args.train_batch_size, args.eval_batch_size)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    evaluate_model(args.dataset_name, args.model_name, tokenizer, args.num_samples)

if __name__ == "__main__":
    main()
