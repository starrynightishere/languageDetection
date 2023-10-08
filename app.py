import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

data = pd.read_csv("/path/to/LanguageDetection.csv")  # Update the path

app = Flask(__name__)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data["Text"],
                                                    data["Language"],
                                                    test_size=0.33,
                                                    stratify=data["Language"],
                                                    random_state=42)

# Vectorize the texts using CountVectorizer
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

# Train the Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html', detected_language="")

@app.route('/detect_language', methods=['POST'])
def detect_language_route():
    text = request.form.get('text')
    
    user_data = cv.transform([text])

    # Make the prediction
    prediction = model.predict(user_data)

    # Calculate the confusion matrix using the entire test dataset
    y_pred = model.predict(X_test)  # Predictions for the entire test dataset
    confusion = confusion_matrix(y_test, y_pred)

    # Generate and print the classification report
    class_report = classification_report(y_test, y_pred)

    # Get the predicted language label
    detected_language = prediction[0]

    # Pass the detected_language, input_text, confusion, and class_report to result.html
    return render_template('result.html', detected_language=detected_language, input_text=text,
                           class_report=class_report)

if __name__ == '__main__':
    app.run()
