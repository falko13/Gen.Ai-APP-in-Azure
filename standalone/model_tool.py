import subprocess
import sys

def install_package(package):
    """
    Installs a Python package using pip.

    Args:
    - package (str): Name of the package to install.
    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Required packages for the script to function.
required_packages = ['torch', 'transformers']

# Dynamically install required packages if they are not already installed.
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"Package '{package}' not found. Installing...")
        install_package(package)

# Import necessary classes from the transformers package after ensuring installation.
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class TextProcessingModel:
    """
    A model for processing text for sentiment analysis or summarization.

    Attributes:
    - model_name (str): The name of the model to use.
    - tokenizer: The tokenizer for the specified model.
    - model: The actual model loaded for processing.
    """

    def __init__(self, model_name):
        """
        Initializes the model with the specified model name.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def predict_sentiment(self, text):
        """
        Predicts the sentiment of the provided text.

        Args:
        - text (str): The input text to analyze.

        Returns:
        - str: The predicted sentiment.
        """
        prompt = self._construct_prompt(text, output_type="sentiment")
        inputs = self.tokenizer(prompt, return_tensors='pt')
        output = self.model.generate(inputs['input_ids'], max_new_tokens=50)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def summarize_text(self, text):
        """
        Summarizes the provided text.

        Args:
        - text (str): The input text to summarize.

        Returns:
        - str: The summary of the text.
        """
        prompt = self._construct_prompt(text, output_type="summarization")
        inputs = self.tokenizer(prompt, return_tensors='pt')
        output = self.model.generate(inputs['input_ids'], max_new_tokens=200)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def _construct_prompt(self, dialogue, output_type):
        """
        Constructs a prompt for the model based on the output type.

        Args:
        - dialogue (str): The dialogue or text to process.
        - output_type (str): Either 'sentiment' or 'summarization'.

        Returns:
        - str: The constructed prompt.
        """
        prompts = {
            "sentiment": '''Provide Sentiment for the following comment/conversation (possible sentiments: Positive, Negative, Neutral):\n\nComment: "{}"\n\nSentiment: ''',
            "summarization": 'Summarize the following conversation.\n\n{}\n\nSummary: '
        }
        return prompts[output_type].format(dialogue)


def main(text="Your default text here", output_type="summarization", model_name='google/flan-t5-base'):
    """
    Main function to run the text processing model with direct parameters.
    
    Args:
    - text (str): Text to process.
    - output_type (str): Either 'sentiment' or 'summarization'.
    - model_name (str): The model to use for processing.
    """
    processor = TextProcessingModel(model_name)

    if output_type == "sentiment":
        result = processor.predict_sentiment(text)
        print("Sentiment:", result)
    elif output_type == "summarization":
        result = processor.summarize_text(text)
        print("Summary:", result)

if __name__ == "__main__":
    # Example calls for direct testing without needing command-line arguments
    # main(text="Australians hold serious concerns about a cashless society despite making few cash payments.", output_type="summarization")
    main(text="I really enjoy sunny weather.", output_type="sentiment")
