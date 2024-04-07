import subprocess
import sys
from pathlib import Path
import os
import pkg_resources

required_packages_versions = {
    'datasets': '2.17.0',
    'torch': '2.0.0',
    'torchdata': '0.6.1',
    'transformers': '4.27.2',
    'evaluate': '0.4.0',
    'rouge_score': '0.1.2',
    'loralib': '0.1.1',
    'peft': '0.3.0'
}



def install_package(package, version):
    """
    Installs or upgrades a Python package using pip to a specific version.

    Args:
    - package (str): Name of the package to install or upgrade.
    - version (str): Specific version of the package to install or upgrade to.
    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}=={version}", "--quiet", "--disable-pip-version-check"])

subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

for package, version in required_packages_versions.items():
    try:
        # Attempt to import the package
        pkg = pkg_resources.get_distribution(package)
        if pkg.version != version:
            # If the version does not match the desired one, upgrade the package
            print(f"Package '{package}' version {pkg.version} found, but version {version} is required. Upgrading...")
            install_package(package, version)
        else:
            print(f"Package '{package}' version {version} is already installed.")
    except pkg_resources.DistributionNotFound:
        # If the package is not found, install it to the specified version
        print(f"Package '{package}' not found. Installing version {version}...")
        install_package(package, version)



from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
import torch
from datasets import load_dataset
import evaluate
from peft import LoraConfig, get_peft_model, TaskType

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

class TextProcessingModel:    
    def __init__(self, model_name='google/flan-t5-base', use_peft=False, compare_models=False):

        self.model_name = model_name
        self.use_peft = use_peft
        print('init:',self.use_peft)
        self.compare_models = compare_models
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_dir = "./models/peft_optimized_model"
        self.model_path = Path(self.model_dir)
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        if self.peft_model_exists() and self.use_peft:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.base_model, 
                                                               self.model_dir,
                                                               torch_dtype=torch.bfloat16,
                                                               is_trainable=False)
        else:
            self.model = self.base_model
            if self.use_peft:
                self.train_peft_model()

    def peft_model_exists(self):
        required_files = ['adapter_model.bin', 'adapter_config.json']
        return all((self.model_path / f).exists() for f in required_files)

    def predict_sentiment(self, text):
        prompt = self._construct_prompt(text, output_type="sentiment")
        inputs = self.tokenizer(prompt, return_tensors='pt')
        output = self.model.generate(inputs['input_ids'], max_new_tokens=50)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
    

    def _construct_prompt(self, dialogue, output_type):
        prompts = {
            "sentiment": f"Provide Sentiment for the following comment/conversation (possible sentiments: Positive, Negative, Neutral):\n\nComment: \"{dialogue}\"\n\nSentiment: ",
            "summarization": f"Summarize the following conversation.\n\n{dialogue}\n\nSummary: "
        }
        return prompts[output_type]

    def train_peft_model(self):
        print("Training PEFT model...")
        dataset_name = 'knkarthick/dialogsum'
        model_name = self.model_name
        output_dir = self.model_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        tokenized_datasets = tokenize_and_prepare_data(dataset_name, model_name)
        
        # Capture the returned trained model
        fine_tune_with_peft(tokenized_datasets, model_name, output_dir)
        # trained_peft_model = fine_tune_with_peft(tokenized_datasets, model_name, output_dir)
        
        
        
        # Now save the PEFT model adaptor and tokenizer
        # trained_peft_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        tokenizer = self.tokenizer

        # Update self.model by applying newly trained PEFT adaptor
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.base_model,self.model_dir,torch_dtype=torch.bfloat16,is_trainable=False)
        print("PEFT model trained and saved successfully.")


    def _construct_prompt(self, dialogue, output_type):
        prompts = {
            "sentiment": f"Provide Sentiment for the following comment/conversation (possible sentiments: Positive, Negative, Neutral):\n\nComment: \"{dialogue}\"\n\nSentiment: ",
            "summarization": f"Summarize the following conversation.\n\n{dialogue}\n\nSummary: "
        }
        return prompts[output_type]
    
    def compare_summaries(self, text):
        if not self.compare_models or not self.peft_model_exists():
            print("Comparison not possible.")
            return
        print("Comparing Base Model and PEFT Model Summaries...")
        base_summary = self.generate_summary(text, self.base_model)
        peft_summary = self.generate_summary(text, self.model)
        print(f"Base Model Summary: {base_summary}\nPEFT Model Summary: {peft_summary}")
        
    def generate_summary(self, text, model=None):
        if not model:
            model = self.model
        prompt = f"Summarize the following conversation:\n{text}\nSummary:"
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = model.generate(inputs['input_ids'], max_length=150, early_stopping=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

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
    print(print_number_of_trainable_model_parameters(peft_model))

    training_args = TrainingArguments(
        output_dir=output_dir,
        auto_find_batch_size=True,
        learning_rate=1e-3, # Higher learning rate than full fine-tuning.
        num_train_epochs=1,
        logging_steps=1,
        max_steps=1,
        report_to="none"  # Set to 'mlflow' to enable mlflow logging
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_datasets["train"]
        # eval_dataset=tokenized_datasets["validation"],
        # compute_metrics=compute_metrics
    )
    print('training')


    trainer.train()

    # Now save the PEFT model adaptor and tokenizer
    trainer.model.save_pretrained(output_dir)
    # return trainer.model

def main(text, output_type="summarization", model_name='google/flan-t5-base', use_peft=False, compare_models=False):
    processor = TextProcessingModel(model_name, use_peft, compare_models)
    if output_type == "sentiment":
        result = processor.predict_sentiment(text)
        print("Sentiment:", result)
    elif output_type == "summarization":
        if compare_models:
            processor.compare_summaries(text)
        else:
            summary = processor.generate_summary(text)
            print(f"Summary: {summary}")

dataset = load_dataset('knkarthick/dialogsum')
# print(dataset)

dialogue = dataset['test'][0]['dialogue']
summary = dataset['test'][0]['summary']

# print(dialogue,'___',summary)

if __name__ == "__main__":
    # Example calls for direct testing without needing command-line arguments
    main(text=dialogue, output_type="summarization", use_peft=True, compare_models=False)

