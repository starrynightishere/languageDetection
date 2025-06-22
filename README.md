# Language Detection Using Character Frequency and Cosine Similarity

This notebook demonstrates a simple, interpretable method to detect the language of a given text using basic text processing and cosine similarity.

### What It Does
- Extracts character frequency vectors from sample texts
- Compares them using cosine similarity
- Predicts the language of unseen text inputs

### Tech Stack
- Python 3
- NumPy
- No external machine learning libraries

### Structure
- `languageDetection.ipynb`: Main notebook with code, tests, and example predictions.

### Example
Input: `"Hola, ¿cómo estás?"`  
Output: `Predicted language: Spanish`

### Note
This is a basic prototype for learning purposes. For real-world usage, consider tools like [langdetect](https://pypi.org/project/langdetect/) or [fastText](https://fasttext.cc/).


Feel free to fork or contribute!
