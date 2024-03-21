# Import libraries
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
dataset = pd.read_csv("Nifty Historical - NSE-NIFTY.csv", skiprows=1)
# Remove leading and trailing spaces from column names
dataset.columns = dataset.columns.str.strip()

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
        self.dataset = dataset

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
    
    def analyze_trends(self, date):
        # Look up data for the specified date
        data_for_date = self.dataset[self.dataset['DATE'] == date]

        if data_for_date.empty:
            return None

        return data_for_date

# Define function to retrieve information based on intent
def get_response(intent, date=None):
    response = ""
    if date:
        while True:
            try:
                trend_row = classifier.analyze_trends(date)
                if trend_row is None or trend_row.empty:
                    response = "Data not available for the specified date. Please try another date."
                else:
                    if intent.lower() == "expected high":
                        response = f"The expected high for {date} is {trend_row['EXP HIGH'].iloc[0]}."
                        
                    elif intent.lower() == "expected low":
                        response = f"The expected low for {date} is {trend_row['EXP LOW'].iloc[0]}."
                    else:
                        response = f"Here are the trends for {date}:\n"
                        response += f"Open: {trend_row['OPEN'].iloc[0]}\n"
                        response += f"Pivot: {trend_row['PIVOT'].iloc[0]}\n"
                        response += f"Expected High: {trend_row['EXP HIGH'].iloc[0]}\n"
                        response += f"Expected Low: {trend_row['EXP LOW'].iloc[0]}\n"
                        response += f"High: {trend_row['ACT HIGH'].iloc[0]}\n"
                        response += f"Low: {trend_row['ACT LOW'].iloc[0]}\n"
                        response += f"Close: {trend_row['CLOSE'].iloc[0]}\n"
                        
                break
            except ValueError:
                response = "Please enter a valid date in the format YYYY-MM-DD."
                break
    else:
        response = "Please specify a date to analyze trends."
    return response

# Main function to handle user queries
def chatbot(query):
    # Classify intent
    intent = classifier.classify_intent(query)
    # Get date from query if mentioned
    date = None
    if "on" in query.split() and query.split().index("on") < len(query.split()) - 1:
        date_index = query.split().index("on") + 1
        date = query.split()[date_index]
    # Get response based on intent and date
    response = get_response(intent, date)
    return response

# Instantiate IntentClassifier
classifier = IntentClassifier(dataset)

# Example usage
print("QuantBlu Bot: Hi, I'm a Nifty trend chatbot. Feel free to ask me about Nifty trends!")
while True:
    user_query = input("User: ")
    bot_response = chatbot(user_query)
    print("Bot:", bot_response)
