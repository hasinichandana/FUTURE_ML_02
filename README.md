# Support Ticket Classification & Prioritization System

## Overview

This project is an NLP-based Machine Learning system that automatically classifies customer support tickets into different categories such as:

* Billing Inquiry
* Technical Issue
* Refund Request
* Cancellation Request
* Product Inquiry

The system helps businesses automate support workflows, reduce manual ticket sorting, and improve response efficiency.

---

## Features

* NLP text preprocessing
* Stopword removal
* Lemmatization
* TF-IDF vectorization
* Ticket classification using Machine Learning
* Confusion matrix visualization
* Automated support response generation
* Model saving using joblib
* Real-time ticket prediction

---

## Technologies Used

* Python
* Pandas
* NLTK
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib

---

## Project Structure

## Project Structure

```text id="o6q2vy"
support-ticket-classification/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── tickets.csv
│
├── models/
│   ├── ticket_classifier.pkl
│   └── tfidf_vectorizer.pkl
│
├── screenshots/
│   └── confusion_matrix.png
│
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   └── predict.py
│
└── venv/
```

### Folder & File Description

| File / Folder                      | Description                                  |
| ---------------------------------- | -------------------------------------------- |
| `app.py`                           | Streamlit web application                    |
| `requirements.txt`                 | Project dependencies                         |
| `README.md`                        | Project documentation                        |
| `data/tickets.csv`                 | Customer support ticket dataset              |
| `models/ticket_classifier.pkl`     | Trained ML model                             |
| `models/tfidf_vectorizer.pkl`      | Saved TF-IDF vectorizer                      |
| `screenshots/confusion_matrix.png` | Model evaluation screenshot                  |
| `src/preprocess.py`                | Text preprocessing and cleaning              |
| `src/train_model.py`               | Model training and evaluation                |
| `src/predict.py`                   | Ticket prediction and automated responses    |
| `venv/`                            | Virtual environment (not uploaded to GitHub) |

```
```


---

## Machine Learning Workflow

1. Load support ticket dataset
2. Clean and preprocess text
3. Convert text into numerical vectors using TF-IDF
4. Train classification model
5. Evaluate model performance
6. Predict ticket categories
7. Generate automated support responses

---

## NLP Preprocessing

The following preprocessing techniques were applied:

* Lowercasing
* Punctuation removal
* Number removal
* Stopword removal
* Lemmatization

---

## Model Used

* Multinomial Naive Bayes
* TF-IDF Vectorizer

---

## Evaluation Metrics

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## Live Demo  
https://supportticketclassificationandprioritizationml.streamlit.app/

---
## Business Impact

This system can help organizations:

* Automatically categorize support tickets
* Reduce manual workload
* Improve ticket routing
* Increase support efficiency
* Speed up customer response time

---

## Future Improvements

* Deep learning models
* Better quality datasets
* Priority prediction system
* Streamlit web deployment
* Multi-language support

---

## Conclusion

This project demonstrates the use of NLP and Machine Learning for automating customer support ticket classification and response systems.
