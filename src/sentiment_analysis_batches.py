import mlflow.pyfunc
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class SentimentAnalysisModel(mlflow.pyfunc.PythonModel):
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def predict(self, context, model_input):
        # Ensure model_input is a DataFrame
        if not isinstance(model_input, pd.DataFrame):
            raise ValueError("Model input must be a pandas DataFrame")
        
        results = []
        for index, row in model_input.iterrows():
            dialogue = row['text']
            prompt = self.construct_prompt(dialogue)
            inputs = self.tokenizer(prompt, return_tensors='pt')
            output = self.model.generate(inputs['input_ids'], max_new_tokens=50)
            decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
            results.append(decoded_output)
        
        # Return a DataFrame with the results
        return pd.DataFrame(results, columns=['sentiment'])
    
    def construct_prompt(self, dialogue):
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
        return start_prompt + '"' + dialogue + '"' + end_prompt
