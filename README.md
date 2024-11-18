# Sentiment Analysis Project

This project performs sentiment analysis using machine learning and a TF-IDF vectorizer.

## Files

- `SentimentAnalysis.ipynb`: The main Jupyter Notebook for running and testing the analysis.
- `sentiment_model.pkl`: The trained sentiment analysis model.
- `vectorizer.pkl`: The TF-IDF vectorizer used for text preprocessing.
- `positive_examples.txt`: Sample positive text data for testing.
- `sentimentanalysis.py`: Script for running sentiment analysis on text files.
- `requirements.txt`: List of Python dependencies.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/NehaBL/SentimentAnalysisProject.git
   ## Running the Project

### Jupyter Notebook
To run the Jupyter Notebook:
1. Open the notebook:
   ```bash
   jupyter notebook SentimentAnalysis.ipynb

### **Usage Instructions**

Provide example inputs and outputs so users understand how to use your project.

```markdown
## Usage

1. Enter a sample text in the input field (if using Streamlit).
2. Click the "Analyze Sentiment" button.
3. The app will display whether the sentiment is Positive or Negative.
## Troubleshooting

- **SSL Verification Error for NLTK Downloads**: If you encounter an SSL verification error, try bypassing SSL verification using the code in the "Download NLTK Data" section.
- **FileNotFoundError**: Ensure all files (`sentiment_model.pkl`, `vectorizer.pkl`, etc.) are in the correct directory.

## Examples

Here are some examples of inputs and the corresponding outputs from the sentiment analysis:

- **Input:** "I love this product!"
  - **Output:** "Sentiment: Positive"

- **Input:** "This is the worst experience."
  - **Output:** "Sentiment: Negative"
# Sentiment Analysis Project

This project performs sentiment analysis using machine learning and a TF-IDF vectorizer.

## Project Overview
The aim of this project is to build a machine learning model that can analyze the sentiment of textual data. By using Natural Language Processing (NLP) techniques, this project classifies text into positive and negative sentiments. This is particularly useful for businesses or organizations seeking to understand customer feedback, social media posts, and product reviews at scale.

### Goals and Objectives
- **Train a Sentiment Analysis Model**: Use a dataset of labeled text data to train a machine learning model that can classify text as positive or negative.
- **Deploy for Real-World Usage**: The project includes a deployment-ready setup, allowing the model to be used in a live setting through a Streamlit interface.
- **Optimize Accuracy**: Experiment with different vectorization methods (like TF-IDF) and preprocessing steps to improve the model’s accuracy.



- **Input:** "I feel great today!"
  - **Output:** "Sentiment: Positive"

- **Input:** "This is so disappointing."
  - **Output:** "Sentiment: Negative"
## Methodology

This section explains the steps taken to develop the sentiment analysis model, including data preprocessing, feature extraction, and model training.

### 1. Data Collection and Preparation
The dataset used for training the model contains labeled text data indicating positive or negative sentiment. Each data entry consists of:
- **Text**: The text input (e.g., product reviews, feedback).
- **Label**: The sentiment label, either "positive" or "negative."

### 2. Preprocessing
To improve model accuracy, several preprocessing steps were applied to the text data:
- **Tokenization**: Splitting text into individual words or tokens.
- **Stopword Removal**: Removing common words (e.g., "the," "is") that don’t contribute much to sentiment.
- **Lowercasing**: Converting all text to lowercase to reduce case sensitivity.
- **Punctuation Removal**: Removing punctuation to focus on meaningful words.
- **Lemmatization**: Reducing words to their base form (e.g., "running" to "run").

### 3. Feature Extraction using TF-IDF Vectorization
To convert the text data into a f

