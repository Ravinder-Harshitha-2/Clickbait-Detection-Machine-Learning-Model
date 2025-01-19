import pandas as pd
import joblib

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Preprocess text function
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    tokens = word_tokenize(text.lower())
    processed_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word.isalnum() and word not in stop_words
    ]
    
    return " ".join(processed_tokens)

# Load the dataset
file_path = "clickbait_data.csv"  # Replace with your dataset path
data = pd.read_csv(file_path)

# Assuming 'headline' is the column with the titles and 'label' is the column with clickbait (1) or not (0)
data['processed_headline'] = data['headline'].apply(preprocess_text)

# Split the data
x = data['processed_headline']
y = data['clickbait']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(x_train)
X_test_vec = vectorizer.transform(x_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate the model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Load the trained model and vectorizer
model = joblib.load('clickbait_model.pkl')
vectorizer = joblib.load('count_vectorizer.pkl')

# Function to predict clickbait using both methods
def detect_clickbait(title):
    # Machine learning prediction
    processed_title = preprocess_text(title)
    transformed_title = vectorizer.transform([processed_title])
    prediction = model.predict(transformed_title)
    return "Clickbait" if prediction[0] == 1 else "Not Clickbait"


# Save the model and vectorizer using joblib
joblib.dump(model, 'clickbait_model.pkl')  # Save the trained model
joblib.dump(vectorizer, 'count_vectorizer.pkl')  # Save the vectorizer

