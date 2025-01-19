import pandas as pd
import re
import joblib

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


clickbait_data = pd.read_csv('youtube_clickbait.csv') 
non_clickbait_data = pd.read_csv('youtube_notClickbait.csv')

clickbait_data = clickbait_data.drop(['ID', 'Views', 'Likes' ,'Dislikes', 'Favorites'], axis=1)
non_clickbait_data = non_clickbait_data.drop(['ID', 'Views', 'Likes' ,'Dislikes', 'Favorites'], axis=1)

# Add a label column (1 for clickbait, 0 for non-clickbait)
clickbait_data['label'] = 1
non_clickbait_data['label'] = 0

# Combine the datasets
data = pd.concat([clickbait_data, non_clickbait_data], ignore_index=True)

def preprocess_youtube(text):
    # Remove special characters, numbers, and extra spaces
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Keep only letters and spaces
    text = text.lower()  # Convert to lowercase
    text = text.strip()  # Remove extra spaces
    # Tokenize the text into words
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join tokens back into a single string (optional, depending on your use case)
    processed_text = ' '.join(tokens)

    return processed_text

# Apply preprocessing
data['processed_title'] = data['Video Title'].apply(preprocess_youtube)

# Features (processed titles) and labels
X = data['processed_title']
y = data['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

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
    # Machine learning prediction
    for word in clickbait_keywords:
        if word in title.lower():
            return "Clickbait"
        
    processed_title = preprocess_youtube(title)
    transformed_title = vectorizer.transform([processed_title])
    prediction = model.predict(transformed_title)
    return "Clickbait" if prediction[0] == 1 else "Not Clickbait"

    

# Save the model
joblib.dump(model, 'youtube_model.pkl')
# Save the vectorizer
joblib.dump(vectorizer, 'youtube_vectorizer.pkl')










# # Load the trained model and vectorizer
# model = joblib.load('clickbait_model.pkl')
# vectorizer = joblib.load('count_vectorizer.pkl')



# # Function to predict clickbait using both methods
# def detect_clickbait(title):
#     # Machine learning prediction
#     processed_title = preprocess_text(title)
#     transformed_title = vectorizer.transform([processed_title])
#     prediction = model.predict(transformed_title)
#     return "Clickbait" if prediction[0] == 1 else "Not Clickbait"

# Detect clickbait function
# def detect_clickbait(title, content_type):

#     if content_type == "YouTube":
#         processed_title = preprocess_youtube(title)
#         transformed_title = youtube_vectorizer.transform([processed_title])
#         prediction = youtube_model.predict(transformed_title)
    
#     elif content_type == "Articles":
#         processed_title = preprocess_text(title)
#         transformed_title = vectorizer.transform([processed_title])
#         prediction = model.predict(transformed_title)
    
#     else:
#         raise ValueError("Invalid content type. Choose 'YouTube' or 'Articles'.")

#     return "Clickbait" if prediction[0] == 1 else "Not Clickbait"


# # Predefined keywords for rule-based detection
# clickbait_keywords = {"shocking", "unbelievable", "you won't believe", "amazing", "incredible", "secret", "revealed", "never seen before"}

# # Function to predict clickbait using both methods
# def detect_clickbait(title):
#     # Rule-based detection
#     for word in clickbait_keywords:
#         if word in title.lower():
#             return "Clickbait (Rule-based)"