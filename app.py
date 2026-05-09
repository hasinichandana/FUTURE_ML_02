import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('models/ticket_classifier.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Title
st.title("Support Ticket Classification System")

st.write(
    "Enter a customer support ticket to predict its category."
)

# User input
user_input = st.text_area(
    "Support Ticket"
)

# Predict button
if st.button("Predict"):

    # Transform input
    vector = vectorizer.transform([user_input])

    # Predict category
    prediction = model.predict(vector)[0]

    # Automated responses
    responses = {

        "Billing inquiry":
        "Our billing team will review your issue shortly.",

        "Technical issue":
        "Our technical team is investigating the issue.",

        "Refund request":
        "Your refund request has been received.",

        "Cancellation request":
        "Your cancellation request is being processed.",

        "Product inquiry":
        "Our product support team will assist you."
    }

    # Display prediction
    st.subheader("Predicted Ticket Type")
    st.success(prediction)

    # Display automated reply
    st.subheader("Automated Reply")
    st.info(
        responses.get(
            prediction,
            "Thank you for contacting support."
        )
    )