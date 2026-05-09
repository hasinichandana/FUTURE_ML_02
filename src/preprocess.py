import pandas as pd
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4')

# Load dataset
df = pd.read_csv('../data/tickets.csv')

# Initialize tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):

    text = str(text).lower()

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    # Split words
    words = text.split()

    # Remove stopwords and lemmatize
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Apply cleaning to ticket descriptions
df['cleaned_text'] = df['Ticket Description'].apply(clean_text)

# Show results
print(df[['Ticket Description', 'cleaned_text']].head())