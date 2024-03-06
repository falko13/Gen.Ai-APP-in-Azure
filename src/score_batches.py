
import json
import pandas as pd
import mlflow.pyfunc
import os

# Path where the MLflow model is saved
model_path = "models:/SentimentAnalysisModel/Production"

# Global variable for holding the model
model = None

def init():
    global model
    # Load the MLflow model into the global variable
    model = mlflow.pyfunc.load_model(model_path)

def run(mini_batch):
    # mini_batch is a list of file paths
    results = []
    for file_path in mini_batch:
        # Assuming each file contains data in JSON format
        with open(file_path) as f:
            data = json.load(f)
            # Convert data into DataFrame
            df = pd.DataFrame(data["input_data"]["data"], columns=data["input_data"]["columns"])
            # Perform prediction
            prediction = model.predict(df)
            # Assuming prediction is returned as a list of sentiments
            result = {"sentiment": prediction.tolist()}
            results.append(result)
    
    # Save the results to a file
    output_file_path = os.path.join(os.getenv("AZUREML_BI_OUTPUT_PATH"), "results.json")
    with open(output_file_path, "w") as output_file:
        json.dump(results, output_file)
    
    return output_file_path
