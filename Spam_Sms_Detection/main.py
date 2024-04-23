import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import LabelEncoder
import string

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# Load data
messages = pd.read_csv('Spam_Sms_Detection/messages.csv', encoding="ISO-8859-1")

# Data preprocessing
messages.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
messages.rename(columns={'v1': 'Target', 'v2': 'Text'}, inplace=True)
messages.drop_duplicates(inplace=True)

encoder = LabelEncoder()
messages['Target'] = encoder.fit_transform(messages['Target'])

# Split data into ham and spam datasets
ham_messages = messages[messages['Target'] == 0]
spam_messages = messages[messages['Target'] == 1]

# Save datasets to separate CSV files
ham_messages.to_csv('Spam_Sms_Detection/ham_classified.csv', index=False)
spam_messages.to_csv('Spam_Sms_Detection/Classified_spam.csv', index=False)

print("Files saved: 'Classified_ham.csv' and 'Classified_spam.csv'")

# Text processing function
def process_text(text):
    # Convert to lower case
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

# Apply text processing
messages['Processed_text'] = messages['Text'].apply(process_text)

# Feature Extraction
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(messages['Processed_text'])
y = messages['Target'].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X.toarray(), y, test_size=0.2, random_state=2)

# Model training - Support Vector Machine
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Predictions and evaluations
y_pred = svm_classifier.predict(X_test)
print(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")
print(f"Precision Score: {precision_score(y_test, y_pred)}")
