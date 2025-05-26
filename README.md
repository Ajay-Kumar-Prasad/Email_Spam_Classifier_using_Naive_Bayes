# Email_Spam_Classifier_using_Naive_Bayes
A machine learning web application built with Streamlit to classify emails or SMS messages as Spam or Not Spam using a Naive Bayes model trained on a labeled dataset.

website link:https://ajay-kumar-prasad-email-spam-classifier-using-naive--app-cgd9gw.streamlit.app/

![alt text](image.png)

# Model and Data

Algorithm: Multinomial Naive Bayes
Vectorization: TF-IDF
Dataset: SMS Spam Collection Dataset

![alt text](image-1.png)

# How the Model Works

This project uses a Multinomial Naive Bayes classifier to detect spam in SMS or email messages. Here's a quick overview of the workflow:

1.Preprocessing:

Convert text to lowercase
Tokenize using NLTK's TreebankWordTokenizer
Remove non-alphanumeric words and stopwords
Apply stemming with PorterStemmer

# Spam data
![alt text](image-2.png)
![alt text](image-3.png)

# Ham data
![alt text](image-4.png)
![alt text](image-5.png)

2.Feature Extraction:

Use TF-IDF Vectorizer to convert processed text into numerical vectors that represent word importance.

3.Model:

A trained Multinomial Naive Bayes model classifies the message as Spam or Not Spam based on word probabilities.

4.Prediction:

Input message → Preprocessing → Vectorization → Model Prediction → Output on Streamlit UI

# Results

Model Accuracy:     Achieved ~97% accuracy for Multinomial Naive Bayes model

Used different ML algorithms to train and test the data:-

Model	Accuracy	Precision
Gaussian Naive Bayes	0.8501	0.8483
Multinomial Naive Bayes	0.9739	0.9310
Bernoulli Naive Bayes	0.9652	0.7793
Logistic Regression	0.9778	0.8621
Support Vector Classifier	0.9284	0.7172
K-Nearest Neighbors	0.9043	0.3172
Decision Tree Classifier	0.9217	0.4621
Random Forest Classifier	0.9632	0.7448


# Project Structure
.
├── app.py                  # Streamlit app
├── model.pkl               # Trained Naive Bayes model
├── vectorizer.pkl          # Fitted TF-IDF vectorizer
├── README.md               # Project overview
└── requirements.txt        # Python dependencies
