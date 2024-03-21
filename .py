import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
dataset = pd.read_csv("Nifty Historical - NSE-NIFTY.csv", skiprows = 1)
# Remove leading and trailing spaces from column names
dataset.columns = dataset.columns.str.strip()

# Print the cleaned column names
print(dataset.columns)


# Preprocessing functions
def preprocess_text(text):
    # Tokenization, normalization, etc.
    tokens = word_tokenize(text.lower())
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# Intent classification using TF-IDF and cosine similarity
class IntentClassifier:
    def __init__(self, dataset):
        self.vectorizer = TfidfVectorizer(tokenizer=self.preprocess_text)
        self.intent_vectors = self.vectorizer.fit_transform(dataset['TREND'])
    
    def preprocess_text(self, text):
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return tokens

    def classify_intent(self, query):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.intent_vectors)
        most_similar_idx = similarities.argmax()
        intent = dataset.iloc[most_similar_idx]['TREND']
        return intent

# Define function to retrieve information based on intent
def get_response(intent):
    response = ""
    if intent in dataset['TREND'].values:
        trend_row = dataset.loc[dataset['TREND'] == intent].iloc[0]
        response = f"Here are the details for {trend_row['TREND']}:\n"
        response += f"Date: {trend_row['DATE']}\n"
        response += f"Open: {trend_row['OPEN']}\n"
        response += f"High: {trend_row['ACT HIGH']}\n"
        response += f"Low: {trend_row['ACT LOW']}\n"
        response += f"Close: {trend_row['CLOSE']}\n"
        response += f"Expected High: {trend_row['EXP HIGH']}\n"
        response += f"Expected Low: {trend_row['EXP LOW']}\n"
    else:
        response = "I'm sorry, I don't have information about that trend."
    return response

while True:
    user_input = input("User: ")
    intent = get_intent(user_input)
    response = get_response(intent)
    print("Bot: ", response)

# Main function to handle user queries
def chatbot(query):
    # Classify intent
    intent = classifier.classify_intent(query)
    # Get response based on intent
    response = get_response(intent)
    return response

# Instantiate IntentClassifier
classifier = IntentClassifier(dataset)

# Example usage
user_query = "What's the trend for 2024-01-19?"
bot_response = chatbot(user_query)
print(bot_response)

