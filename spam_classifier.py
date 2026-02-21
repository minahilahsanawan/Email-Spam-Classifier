import pandas as pd
from sklearn.model_selection import train_test_split
# converts text into numeric features (TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer
# good for spam/ham with well-behaved probabilities
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# for basic text cleaning (stopwords + stemming)
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
# for removing unwanted characters using patterns
import re
# for saving/loading the trained model and vectorizer
import pickle
import os
# files created after training
mf= "spam_classifier_model.pkl"
vf= "tfidf_vectorizer.pkl"
datasetfile= "spam_ham_dataset.csv"
# decision threshold (increase to reduce false positives, decrease to catch more spam)
SPAM_THRESHOLD = 0.65
# download NLTK stopwords (only if missing)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
class SpamClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.is_trained = False
    def preprocess_text(self, text: str) -> str:
        text = str(text).lower()
        # keep letters + digits; remove other characters
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        words = text.split()
        # stopwords
        try:
            sw = set(stopwords.words("english"))
            words = [w for w in words if w not in sw]
        except Exception:
            words = [w for w in words if len(w) > 2]
        # stemming
        try:
            stemmer = PorterStemmer()
            words = [stemmer.stem(w) for w in words]
        except Exception:
            pass
        return " ".join(words)
    def train_model(self) -> float:
        if not os.path.exists(datasetfile):
            raise FileNotFoundError(f"Dataset file '{datasetfile}' not found.")
        df = pd.read_csv(datasetfile)
        if "Unnamed: 0" in df.columns:
            df = df.drop("Unnamed: 0", axis=1)
        print("Preprocessing emails...")
        df["processed_text"] = df["text"].apply(self.preprocess_text)
        print("Extracting features...")
        self.vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        X = self.vectorizer.fit_transform(df["processed_text"])
        y = df["label_num"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print("Training model...")
        self.model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Model trained with accuracy: {acc:.2f}")
        print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)
        self.is_trained = True
        return float(acc)
    def predict(self, email_text: str) -> dict:
        if not self.is_trained:
            raise ValueError("Sorry, Model not trained yet.")
        processed = self.preprocess_text(email_text)
        vec = self.vectorizer.transform([processed])
        proba = self.model.predict_proba(vec)[0]
        ham_p = float(proba[0])
        spam_p = float(proba[1])
        # threshold decision (reduces borderline false positives)
        prediction = "Spam" if spam_p >= SPAM_THRESHOLD else "Ham"
        confidence = spam_p if prediction == "Spam" else ham_p
        return{"prediction": prediction, "confidence": float(confidence), "probabilities": {"Ham": ham_p, "Spam": spam_p}, "threshold": float(SPAM_THRESHOLD)}
    def save_model(self) -> None:
        if not self.is_trained:
            raise ValueError("Sorry, Model not trained yet.")
        with open(mf, "wb") as f:
            pickle.dump(self.model, f)
        with open(vf, "wb") as f:
            pickle.dump(self.vectorizer, f)
        print(f"Model saved to {mf} and {vf}")
    def load_model(self) -> bool:
        if not os.path.exists(mf) or not os.path.exists(vf):
            return False
        try:
            with open(mf, "rb") as f:
                self.model = pickle.load(f)
            with open(vf, "rb") as f:
                self.vectorizer = pickle.load(f)
            self.is_trained = True
            return True
        except Exception:
            return False
classifier = SpamClassifier()
def main():
    print("Email Spam Classifier (Spam vs. Ham)")
    if classifier.load_model():
        print("Loaded existing model!!")
    else:
        print("Training new model...")
        classifier.train_model()
        classifier.save_model()
    test_email = "Subject: Interview confirmation. Please arrive 10 minutes early!"
    r = classifier.predict(test_email)
    print("Test Email:", test_email)
    print("Prediction:", r["prediction"])
    print("Confidence:", round(r["confidence"], 3))
    print("Probabilities:", r["probabilities"])
    print("Threshold:", r["threshold"])
if __name__ == "__main__":
    main() 