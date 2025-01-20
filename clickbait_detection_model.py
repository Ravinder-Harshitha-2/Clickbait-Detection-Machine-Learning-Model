# Importing Necessary Libraies
import pandas as pd
import joblib
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# Preprocess text function for processing news headlines
def preprocess_news(text):
    # Lemmatizeing the tokens
    lemmatizer = WordNetLemmatizer()
    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    # Converting to lowercase
    tokens = word_tokenize(text.lower())
    processed_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word.isalnum() and word not in stop_words
    ]
  
    return " ".join(processed_tokens)

# Loading the news headlines clickbait dataset
file_path = "clickbait_data.csv"  
data = pd.read_csv(file_path)

# Applying the preprocess news function to 'headline' column in the dataset
data['processed_headline'] = data['headline'].apply(preprocess_news)

# Spliting the data
x = data['processed_headline']
y = data['clickbait']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Vectorizeing the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(x_train)
X_test_vec = vectorizer.transform(x_test)

# Training and fitting the model with Naive Bayes algorithm
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluating the model with accuracy score
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"News Headlines Model Accuracy: {accuracy * 100:.2f}%")

# Evaluating the model with classification report
report = classification_report(y_test, y_pred)
print("News Headlines Classification Report:\n", report)

# Function to predict clickbait 
def detect_clickbait(title):
    # Machine learning prediction
    processed_title = preprocess_news(title)
    transformed_title = vectorizer.transform([processed_title])
    prediction = model.predict(transformed_title)
    return "Clickbait" if prediction[0] == 1 else "Not Clickbait"


# Save the model and vectorizer using joblib
joblib.dump(model, 'clickbait_model.pkl')  
joblib.dump(vectorizer, 'count_vectorizer.pkl')  



#---------------------------------------------------------------------------------------------------------------------


# Loading the Youtube clickbait dataset
clickbait_data = pd.read_csv('youtube_clickbait.csv') 
non_clickbait_data = pd.read_csv('youtube_notClickbait.csv')

# Droping the unnecessary colcumns from each .csv files
clickbait_data = clickbait_data.drop(['ID', 'Views', 'Likes' ,'Dislikes', 'Favorites'], axis=1)
non_clickbait_data = non_clickbait_data.drop(['ID', 'Views', 'Likes' ,'Dislikes', 'Favorites'], axis=1)

# Adding a label column denoting 1 for clickbait, 0 for non-clickbait
clickbait_data['label'] = 1
non_clickbait_data['label'] = 0

# Combineing the two .csv files
data = pd.concat([clickbait_data, non_clickbait_data], ignore_index=True)

# Preprocess text function for processing youtube titles
def preprocess_youtube(text):
    # Removing special characters, numbers, and extra spaces
    text = re.sub(r'[^A-Za-z\s]', '', text)  
    # Converting to lowercase
    text = text.lower()  
    # Removing extra spaces
    text = text.strip()  
    # Tokenizeing the text into words
    tokens = word_tokenize(text)

    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatizeing the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join tokens back into a single string 
    processed_text = ' '.join(tokens)
    return processed_text

# Applying preprocessing to 'Video Title' column in the dataset
data['processed_title'] = data['Video Title'].apply(preprocess_youtube)

# Spliting the data
X = data['processed_title']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizeing the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Training and fitting the model with Naive Bayes algorithm
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluating the model with accuracy score
Y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, Y_pred)
print(f"Youtube Title Model Accuracy: {accuracy * 100:.2f}%")

# Evaluating the model with classification report
report = classification_report(y_test, Y_pred)
print("Youtube Title Classification Report:\n", report)

# Predefined keywords for rule-based detection
clickbait_keywords = {"shocking", 
                      "unbelievable", 
                      "you won't believe", 
                      "amazing", 
                      "incredible", 
                      "secret", 
                      "revealed", 
                      "never seen before",
                      "Leading expert reveals the biggest secret…",
                      "like this",
                      "just now",
                      "things will change your life",
                      "STORYTIME (not clickbait)",
                      "things you must know",
                      "Find out what happened",
                      "be careful"}

# Function to predict clickbait using both methods
def detect_youtube_clickbait(title):
    # rule-based detection
    for word in clickbait_keywords:
        if word in title.lower():
            return "Clickbait"
        
    # Machine learning prediction    
    processed_title = preprocess_youtube(title)
    transformed_title = vectorizer.transform([processed_title])
    prediction = model.predict(transformed_title)
    return "Clickbait" if prediction[0] == 1 else "Not Clickbait"

    
# Save the model and vectorizer using joblib
joblib.dump(model, 'youtube_model.pkl')
joblib.dump(vectorizer, 'youtube_vectorizer.pkl')