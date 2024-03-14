# Gen.Ai-APP-in-Azure

## Overview

`gen.ai` leverages advanced AI models for sentiment analysis and dialogue summarization, focusing on processing and understanding dialogues. The project is currently in progress, with components under development and not yet optimized for final use. The aim is to provide robust solutions for natural language processing tasks using state-of-the-art AI techniques.

## Project Status

:warning: **Project In Progress** - The components provided are in development and may undergo significant changes. They are not yet optimized for production. The Azure version for sentiment analysis has been completed, demonstrating effective sentiment analysis capabilities. Further development will focus on enhancing the sentiment model itself in the General Python version.

## Versions

### General Python Version

Under further development, this version can run independently of any cloud services, demonstrating the core functionality of sentiment analysis and dialogue summarization through standalone scripts. Future enhancements will aim to improve the sentiment model's accuracy and efficiency.

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
- **Tests**: Unit tests for scripts in `../tests`.
- **Components**: Azure ML component defined in `sentiment_prediction.yml`.

## Future Work

- Further refinement of the sentiment prediction component for enhanced accuracy in the General Python version.
- Development and testing of the **Dialogue Summarization with PEFT** component.
- Anticipated development and integration of additional NLP components and features.

**Note:** This README will be updated as the project evolves to reflect new developments, enhancements, and the addition of further components.
