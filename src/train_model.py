import pandas as pd
import re
import string
import nltk
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

# Download NLTK data
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4')

# Load dataset
df = pd.read_csv('../data/tickets.csv')

# Initialize preprocessing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'\d+', '', text)

    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Clean ticket descriptions
# Combine subject + description
df['combined_text'] = (
    df['Ticket Subject'].astype(str)
    + " "
    + df['Ticket Description'].astype(str)
)

# Clean combined text
df['cleaned_text'] = df['combined_text'].apply(clean_text)

# Features (X)
X = df['cleaned_text']

# Target (y)
y = df['Ticket Type']

# Convert text into TF-IDF vectors
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2)
)

X_vectorized = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)

# Detailed report
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))





# Create confusion matrix
cm = confusion_matrix(y_test, predictions)

# Bigger figure size
plt.figure(figsize=(10,8))

# Plot heatmap
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

# Labels and title
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title("Confusion Matrix", fontsize=14)

# Rotate labels for visibility
plt.xticks(rotation=20)
plt.yticks(rotation=0)

# Automatically adjust layout
plt.tight_layout()

# Save image
plt.savefig('../screenshots/confusion_matrix.png')

# Show plot
plt.show()



# Save trained model
joblib.dump(model, '../models/ticket_classifier.pkl')

# Save TF-IDF vectorizer
joblib.dump(vectorizer, '../models/tfidf_vectorizer.pkl')

print("Model and vectorizer saved successfully!")