# Email Spam Classifier (Spam vs Ham)

A machine learning project that classifies emails as Spam or Ham. The pipeline preprocesses raw email text, extracts TF IDF features, and uses a trained classifier to predict the label with probability based confidence. A Flask web interface provides real time classification through a simple form and a JSON API.
<img width="954" height="826" alt="image" src="https://github.com/user-attachments/assets/cc7945de-0c23-48b5-a29f-52439de0f0a4" />

---
## Features

1. Text preprocessing (lowercasing, cleanup, stopword removal, stemming)
2. Feature extraction using TF IDF
3. Binary classification (Spam vs Ham)
4. Confidence and class probabilities returned for each prediction
5. Flask web UI (HTML form) plus REST API endpoint
6. Model persistence using saved artifacts (`.pkl` files)
---
## Methodology

The classification workflow follows a standard text classification pipeline:

1. **Input Handling**: Email subject/body text is provided through the web form.
2. **Preprocessing**: The text is normalized and cleaned to reduce noise and improve feature quality.
3. **Vectorization**: TF-IDF converts the cleaned text into numerical feature vectors.
4. **Prediction**: The trained classifier predicts Spam or Ham and returns probabilities used as confidence.
5. **Serving Layer**: Flask routes handle UI rendering and API responses.
---
## Tech Stack

1. Python
2. scikit learn
3. pandas, numpy
4. NLTK (stopwords)
5. Flask (web interface)

## Repository Structure
### Email-Spam-Classifier
1. app.py
2. spam_classifier.py
3. requirements.txt
4. run.bat
5. spam_classifier_model.pkl
6. tfidf_vectorizer.pkl
7. templates/
8. index.html
---
## Setup and Run (Windows)

### One-Click Run

1. Download or clone this repository.
2. Open the project folder.
3. Double click `run.bat`.
4. When the server starts, open the web app:
   `http://127.0.0.1:5000`
---
## Usage

1. Paste the email subject and body into the text box.
2. Click **Analyze Email**.
3. The system returns:
   - Predicted label: **Spam** or **Ham**
   - Model confidence (%)
   - Ham vs Spam probability bars
---

## Author

Minahil Ahsan Awan   
Email: minahilahsaanawan@gmail.com  
LinkedIn: https://linkedin.com/in/minahilahsaanawan


