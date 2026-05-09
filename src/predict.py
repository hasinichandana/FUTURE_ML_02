import joblib

# Load model and vectorizer
model = joblib.load('../models/ticket_classifier.pkl')

vectorizer = joblib.load('../models/tfidf_vectorizer.pkl')

# User input
sample_ticket = [
    "I want refund because my payment was deducted twice"
]

# Convert text into vector
sample_vector = vectorizer.transform(sample_ticket)

# Predict category
prediction = model.predict(sample_vector)[0]

print("Predicted Ticket Type:", prediction)

# Auto replies based on category
responses = {

    "Billing inquiry":
    "Our billing team will review your payment issue shortly.",

    "Technical issue":
    "Our technical support team is investigating your issue.",

    "Refund request":
    "Your refund request has been received and is being processed.",

    "Cancellation request":
    "Your cancellation request has been forwarded to the support team.",

    "Product inquiry":
    "Our product support team will provide the requested information soon."
}

# Print automated response
print("Automated Reply:")
print(responses.get(
    prediction,
    "Thank you for contacting support."
))