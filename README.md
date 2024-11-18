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

### 6. **Usage Instructions**

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

- **Input:** "I feel great today!"
  - **Output:** "Sentiment: Positive"

- **Input:** "This is so disappointing."
  - **Output:** "Sentiment: Negative"
