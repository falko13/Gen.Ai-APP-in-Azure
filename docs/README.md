# Gen.Ai-APP-in-Azure

## Overview

`gen.ai` leverages advanced AI models for sentiment analysis and dialogue summarization, focusing on processing and understanding dialogues. The project is currently in progress, with components that are under development and not yet optimized for final use. The aim is to provide robust solutions for natural language processing tasks using state-of-the-art AI techniques.

## Project Status

:warning: **Project In Progress** - The components provided are in development and may undergo significant changes. They are not yet optimized for production. The sentiment analysis component has been tested and deployed as a component in Azure within a pipeline workflow, with plans for further improvement of sentiment detection accuracy.

## Versions

There are two main versions of this project:

### General Python Version

This version can run independently of any cloud services, demonstrating the core functionality of sentiment analysis and dialogue summarization through standalone scripts.

### Azure Python SDK 2.0 Version

Developed for the Azure Python SDK 2.0, this version utilizes the latest features and improvements as of October 2023. It aims to leverage Azure Machine Learning components and pipelines for scalable and efficient processing of natural language tasks.

## Components

The project currently includes the following components, designed for comprehensive natural language processing:

1. **Sentiment Prediction**: This component, which integrates tokenization and sentiment analysis into a single step, predicts the sentiment of dialogues. It has been successfully tested and deployed within an Azure pipeline workflow, demonstrating effective sentiment analysis capabilities.
   
2. **Dialogue Summarization with PEFT**: A new component fine-tuning a pre-trained model on dialogue summarization tasks using Parameter Efficient Fine-Tuning (PEFT). This component is capable of generating concise summaries from extended dialogues, enhancing the comprehension of conversational content. It's particularly effective with specific communications datasets and can be adapted to accommodate company-specific communication patterns or terminology, producing personalized LLM models.

## Future Work

- Plans are in place for further refinement and optimization of the sentiment prediction component to enhance accuracy.
- The **Dialogue Summarization with PEFT** component is in the early stages of integration and requires further development and testing. This feature demonstrates the potential to expand the project's capabilities significantly.
- The development and integration of additional components and features to cover more aspects of natural language processing are anticipated.

---

**Note:** This README will be updated as the project evolves to reflect new developments, enhancements, and the addition of further components.
