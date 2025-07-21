from flask import Flask, request, render_template, session
import joblib, json, re
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = "your-secret-key"

# Load model files
model = joblib.load("model/model.joblib")
vectorizer = joblib.load("model/vectorizer.joblib")
encoder = joblib.load("model/label_encoder.joblib")

# Load solution bank
with open("data/solution_bank.json", "r") as f:
    solution_bank = json.load(f)

# Load dataset for fallback
df = pd.read_csv("data/multi_app_support_tickets.csv")
df["cleaned"] = df["ticket_text"].str.lower().str.replace(r'[^\w\s]', '', regex=True)
ticket_vectors = vectorizer.transform(df["cleaned"])

@app.route("/", methods=["GET", "POST"])
def index():
    response = None
    solution = None
    question = None

    if request.method == "POST":
        user_input = request.form["user_input"].strip()
        user_input_lower = user_input.lower()

        # First-time input
        if "yes" not in user_input_lower and "no" not in user_input_lower:
            session.clear()

            # Try predict tag
            X_input = vectorizer.transform([user_input_lower])
            tag_num = model.predict(X_input)[0]
            predicted_tag = encoder.inverse_transform([tag_num])[0]

            # Try extract app
            known_apps = ["WhatsApp", "Instagram", "Chrome", "YouTube", "Gmail", "Facebook", "Twitter", "Zoom", "Spotify", "Snapchat", "Netflix", "Telegram"]
            def extract_app_name(text):
                for app in known_apps:
                    if re.search(rf"\b{app.lower()}\b", text.lower()):
                        return app
                return None

            app_found = extract_app_name(user_input)

            # Default error
            fallback_used = False
            key = f"{app_found}::{predicted_tag}" if app_found else None

            if not app_found or key not in solution_bank:
                # Use fallback
                sims = cosine_similarity(X_input, ticket_vectors)[0]
                top_idx = sims.argmax()
                fallback_app = df.iloc[top_idx]["app_name"]
                fallback_tag = df.iloc[top_idx]["tag"]
                key = f"{fallback_app}::{fallback_tag}"
                predicted_tag = fallback_tag
                app_found = fallback_app
                fallback_used = True

            session["app"] = app_found
            session["tag"] = predicted_tag
            session["index"] = 0
            session["solutions"] = solution_bank.get(key, [])

            if session["solutions"]:
                solution = session["solutions"][0]
                question = "Did that help? (yes / no)"
                response = f"📱App: {app_found}<br>❌Issue: {predicted_tag}"
            else:
                response = "😢 Sorry! I couldn’t find a solution for that."

        elif "no" in user_input_lower:
            idx = session.get("index", 0) + 1 
            session["index"] = idx
            sols = session.get("solutions", [])
            if idx < len(sols):
                solution = sols[idx]
                question = "Did that help? (yes / no)"
            else:
                solution = "😔 I’ve given all I could. Please try the official support page."

        elif "yes" in user_input_lower:
            solution = "I'm so glad I could help! Let me know if there's anything else."

    return render_template("index.html", response=response, solution=solution, question=question)

if __name__ == "__main__":
    app.run(debug=True)
