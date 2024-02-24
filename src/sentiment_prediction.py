import mlflow
import argparse
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def predict_sentiment(dialogue, output_path, model_name='google/flan-t5-base'):
    # Initialize tokeniazer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Constructing a 5-shot prompt with examples
    start_prompt = '''Provide Sentiment for the following comment/conversation (possible sentiments: Positive, Negative, Neutral):

    Comment: "I love sunny days, they make me feel so happy!"
    Sentiment: Positive

    Comment: "This is the worst experience of my life, I'm so disappointed."
    Sentiment: Negative

    Comment: "I'm not sure how I feel about this new policy. It might be good or bad."
    Sentiment: Neutral

    Comment: "The service at this restaurant was fantastic, best dinner ever!"
    Sentiment: Positive

    Comment: "I waited for an hour and my order was still wrong."
    Sentiment: Negative

    Comment: '''
    
    end_prompt = '\nSentiment: '
    
    # Construct the full prompt with the user-provided dialogue
    prompt = start_prompt + '"' + dialogue + '"' + end_prompt 

    # Tokenize input dialogue
    inputs = tokenizer(prompt, return_tensors='pt')

    # Generate prediction
    output = model.generate(inputs['input_ids'], max_new_tokens=50)
    
    # Decode and print the prediction
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Write the predicted sentiment to the specified output file
    with open(output_path, 'w') as f:
        f.write('Text: ' + dialogue + '\nPredicted Sentiment: ' + decoded_output + '\n')

def main():
    # enable autologging
    mlflow.autolog()
    
    parser = argparse.ArgumentParser(description="Predict sentiment from input dialogue")
    parser.add_argument("--dialogue", type=str, required=True, help="Input dialogue for sentiment prediction")
    parser.add_argument("--output", type=str, required=True, help="Output file path for sentiment prediction")
    
    args = parser.parse_args()

    # Predict sentiment and write to output
    predict_sentiment(args.dialogue, args.output)

if __name__ == "__main__":
    main()
