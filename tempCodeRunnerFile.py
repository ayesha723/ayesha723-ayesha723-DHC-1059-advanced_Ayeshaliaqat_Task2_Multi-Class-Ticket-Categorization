import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import joblib, os

df = pd.read_csv("data/multi_app_support_tickets.csv")
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["ticket_text"])
encoder = LabelEncoder()
y = encoder.fit_transform(df["tag"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = MultinomialNB()
model.fit(X_train, y_train)
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.joblib")
joblib.dump(vectorizer, "model/vectorizer.joblib")
joblib.dump(encoder, "model/label_encoder.joblib")
print("✅ Model trained and saved!")