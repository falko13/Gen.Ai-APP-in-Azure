import json
import pandas as pd
import mlflow.pyfunc

# Path where the MLflow model is saved, adjust as necessary
model_path = "models:/SentimentAnalysisModel/Production"

# Global variable for holding the model
model = None

def init():
    global model
    # Load the MLflow model into the global variable
    model = mlflow.pyfunc.load_model(model_path)

def run(raw_data):
    try:
        # Parse the incoming JSON data
        input_data = json.loads(raw_data)
        
        # Extract the 'text' value to conform with the input signature
        # Creating a DataFrame to match the expected input format for the predict function
        data = pd.DataFrame({"text": [input_data['text']]})
        
        # Use the global model to predict the sentiment
        prediction = model.predict(data)
        
        # Format the output to match the output signature
        # Assuming prediction is returned as a string
        result = {"sentiment": prediction[0]}
        
        # Return the prediction as JSON
        return json.dumps(result)
    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})
