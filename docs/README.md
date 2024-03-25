# Gen.Ai-APP-in-Azure

## Overview

`gen.ai` leverages advanced AI models for sentiment analysis and dialogue summarization, focusing on processing and understanding dialogues. The project is currently in progress, with components under development and not yet optimized for final use. The aim is to provide robust solutions for natural language processing tasks using state-of-the-art AI techniques.

## Project Status

:warning: **Project In Progress** - The components provided are in development and may undergo significant changes. They are not yet optimized for production. The Azure version for sentiment analysis has been completed, demonstrating effective sentiment analysis capabilities. The General Python Version now includes foundational scripts for both sentiment analysis and dialogue summarization, with the current focus being on establishing a working script example and foundation rather than optimizing accuracy.

## Versions

### General Python Version

This version can now run independently of any cloud services, providing foundational scripts for sentiment analysis and dialogue summarization. The current stage aims at creating a working script example and foundation with the addition of a new standalone script, `model_tool.py`, in the `standalone` folder at the root of the project. Future enhancements will aim to refine these standalone scripts further and may also focus on improving the models' accuracy and efficiency.

### Azure Python SDK 2.0 Version

Developed for the Azure Python SDK 2.0, this version utilizes the latest features and improvements as of October 2023. It aims to leverage Azure Machine Learning components and pipelines for scalable and efficient processing of natural language tasks. The sentiment analysis component in this version has been successfully deployed and is fully operational within an Azure pipeline workflow.

## Components

### Sentiment Prediction

This component integrates tokenization and sentiment analysis into a single step, predicting the sentiment of dialogues. It has been successfully tested and deployed within an Azure pipeline workflow. The sentiment model training, PyFunc wrapping for batch and online inference, and the creation of endpoints and deployments are fully documented and traceable in the main Jupyter notebook.

### Dialogue Summarization with PEFT

Under further development, this new component fine-tunes a pre-trained model on dialogue summarization tasks using Parameter Efficient Fine-Tuning (PEFT). Capable of generating concise summaries from extended dialogues, it enhances the comprehension of conversational content.

## File Structure and Usage

- **Notebooks**: `llm_text_sentiment_main.ipynb` details the full process including model training, inference wrapping, and deployment setup for Azure endpoints.
- **Output**: Located in `../output`, containing example outputs from test components and pipelines.
- **Source Code**: Key scripts in `../src`:
  - `sentiment_prediction.py` for model training and prediction.
  - `sentiment_analysis.py` and `sentiment_analysis_batches.py` for online and batch inference.
  - `score.py` and `score_batches.py` for scoring scripts.
  - `conda-env-mini.yml` and `conda-env.yml` for environment configurations.
- **Standalone**: New `model_tool.py` script added to the `standalone` folder for direct script execution without Azure dependencies.
- **Tests**: Unit tests for scripts in `../tests`.
- **Components**: Azure ML component defined in `sentiment_prediction.yml`.

## Future Work

- Refinement of the standalone scripts in the General Python version for sentiment analysis and dialogue summarization.
- Development and testing of the **Dialogue Summarization with PEFT** component.
- Anticipated development and integration of additional NLP components and features.

**Note:** This README will be updated as the project evolves to reflect new developments, enhancements, and the addition of further components.
