import subprocess
import sys
from pathlib import Path
import os

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_packages = ['torch', 'transformers', 'datasets', 'evaluate', 'mlflow', 'peft']

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"Package '{package}' not found. Installing...")
        install_package(package)

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
import torch
from datasets import load_dataset
import evaluate
from peft import LoraConfig, get_peft_model, TaskType

class TextProcessingModel:
    def __init__(self, model_name='google/flan-t5-base'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_dir = "./models/peft_optimized_model"
        self.model_path = Path(self.model_dir)
        if self.model_path.exists():
            print("Loading PEFT optimized model...")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir)
        else:
            print("Base model loaded. PEFT optimized model can be trained if required.")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def predict_sentiment(self, text):
        prompt = self._construct_prompt(text, output_type="sentiment")
        inputs = self.tokenizer(prompt, return_tensors='pt')
        output = self.model.generate(inputs['input_ids'], max_new_tokens=50)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def summarize_text(self, text, use_peft=False):
        if use_peft and not self.model_path.exists():
            print("Training PEFT model. This might take a while...")
            self.train_peft_model()
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir)
        prompt = self._construct_prompt(text, output_type="summarization")
        inputs = self.tokenizer(prompt, return_tensors='pt')
        output = self.model.generate(inputs['input_ids'], max_new_tokens=200)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def _construct_prompt(self, dialogue, output_type):
        prompts = {
            "sentiment": f"Provide Sentiment for the following comment/conversation (possible sentiments: Positive, Negative, Neutral):\n\nComment: \"{dialogue}\"\n\nSentiment: ",
            "summarization": f"Summarize the following conversation.\n\n{dialogue}\n\nSummary: "
        }
        return prompts[output_type]

    def train_peft_model(self):
        dataset_name = 'knkarthick/dialogsum'
        model_name = self.model_name
        output_dir = self.model_dir
        tokenized_datasets = tokenize_and_prepare_data(dataset_name, model_name)
        fine_tune_with_peft(tokenized_datasets, model_name, output_dir)

def tokenize_and_prepare_data(dataset_name, model_name):
    dataset = load_dataset(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(example):
        start_prompt = 'Summarize the following conversation.\n\n'
        end_prompt = '\n\nSummary: '
        example['input_ids'] = tokenizer([start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]],
                                         padding="max_length", truncation=True, max_length=512, return_tensors="pt").input_ids.squeeze()
        example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True, max_length=128, return_tensors="pt").input_ids.squeeze()
        return example

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["dialogue", "id", "summary", "topic"])
    return tokenized_datasets

def fine_tune_with_peft(tokenized_datasets, model_name, output_dir):
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
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=1e-3,
        num_train_epochs=1,
        # weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
        # greater_is_better=False,
        report_to="none"  # Set to 'mlflow' to enable mlflow logging
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(output_dir)
    print("PEFT model trained and saved successfully.")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    rouge_score = evaluate.load('rouge')
    results = rouge_score.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {"rouge": results}

def main(text="Your default text here", output_type="summarization", model_name='google/flan-t5-base', use_peft=False):
    processor = TextProcessingModel(model_name)

    if output_type == "sentiment":
        result = processor.predict_sentiment(text)
        print("Sentiment:", result)
    elif output_type == "summarization":
        result = processor.summarize_text(text, use_peft=use_peft)
        print("Summary:", result)

if __name__ == "__main__":
    main(text="The recent advancements in AI and machine learning have greatly improved the efficiency and capabilities of natural language processing tasks.", output_type="summarization", use_peft=True)

