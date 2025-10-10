import re
import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import validators
from urllib.parse import urlparse

MODEL_PATH = "model.pkl"

app = Flask(__name__)

# --------------------------
# Feature extraction
# --------------------------
IP_PATTERN = re.compile(
    r'^(?:http[s]?://)?'  # optional scheme
    r'(?:(?:\d{1,3}\.){3}\d{1,3})'  # IPv4
    r'(?::\d+)?(?:/.*)?$'
)

def has_ip(url: str) -> int:
    return int(bool(IP_PATTERN.match(url)))

def count_digits(s: str) -> int:
    return sum(c.isdigit() for c in s)

def url_length(url: str) -> int:
    return len(url)

def count_subdomains(url: str) -> int:
    try:
        host = urlparse(url).netloc
        if host == "":
            host = urlparse("http://" + url).netloc
        # remove port
        host = host.split(':')[0]
        # split on dots and ignore TLD+domain last two
        parts = host.split('.')
        return max(0, len(parts) - 2)
    except Exception:
        return 0

def count_char(url: str, ch: str) -> int:
    return url.count(ch)

def has_https_token(url: str) -> int:
    # sometimes attackers use "https-" in domain to trick
    netloc = urlparse(url).netloc
    return int('https' in netloc.lower())

def is_valid_url(url: str) -> bool:
    # using validators package for basic check
    return validators.url(url)

def extract_features(url: str):
    u = url.strip()
    f = {}
    f['length'] = url_length(u)
    f['count_digits'] = count_digits(u)
    f['count_dots'] = count_char(u, '.')
    f['count_slash'] = count_char(u, '/')
    f['count_at'] = count_char(u, '@')
    f['count_dash'] = count_char(u, '-')
    f['count_question'] = count_char(u, '?')
    f['count_equals'] = count_char(u, '=')
    f['count_amp'] = count_char(u, '&')
    f['count_percent'] = count_char(u, '%')
    f['has_ip'] = has_ip(u)
    f['subdomains'] = count_subdomains(u)
    f['has_https_token'] = has_https_token(u)
    f['valid'] = int(is_valid_url(u))
    return np.array([f[k] for k in sorted(f.keys())], dtype=float), list(sorted(f.keys()))

# --------------------------
# Demo training dataset
# --------------------------
def demo_dataset():
    # Small synthetic example dataset. Replace with a real dataset for production.
    legit = [
        "https://www.google.com/search?q=openai",
        "https://github.com/",
        "https://en.wikipedia.org/wiki/Phishing",
        "https://www.bankofamerica.com/",
        "https://news.ycombinator.com/"
    ]
    phishing = [
        "http://192.168.0.10/login",                         # ip address
        "http://bank-login.example.com/secure",               # suspicious subdomain
        "http://secure-account-verify.com/login?user=1",      # hyphens and login path
        "http://paypal.com-login.verify.suspiciousdomain.net",# deceptive domain
        "http://update-verify.com/secure?account=bank"        # suspicious keywords
    ]
    data = []
    labels = []
    for url in legit:
        feats, _ = extract_features(url)
        data.append(feats)
        labels.append(0)
    for url in phishing:
        feats, _ = extract_features(url)
        data.append(feats)
        labels.append(1)
    X = np.vstack(data)
    y = np.array(labels)
    colnames = sorted({
        'length','count_digits','count_dots','count_slash','count_at','count_dash',
        'count_question','count_equals','count_amp','count_percent','has_ip',
        'subdomains','has_https_token','valid'
    })
    return pd.DataFrame(X, columns=colnames), y

# --------------------------
# Model creation / loading
# --------------------------
def build_model():
    # Simple pipeline: scaler + logistic regression
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(solver='liblinear'))
    ])
    return pipe

def train_and_save_model(path=MODEL_PATH):
    X, y = demo_dataset()
    model = build_model()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    joblib.dump((model, list(X.columns)), path)
    print(f"Trained demo model and saved to {path}. Demo accuracy on holdout: {acc:.2f}")
    return model, list(X.columns)

def load_or_train_model(path=MODEL_PATH):
    if os.path.exists(path):
        try:
            model, columns = joblib.load(path)
            print(f"Loaded model from {path}")
            return model, columns
        except Exception as e:
            print("Failed loading model, retraining. Error:", e)
    return train_and_save_model(path)

model, feature_names = load_or_train_model()

# --------------------------
# Flask routes
# --------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    url = request.form.get("url", "").strip()
    if not url:
        return render_template("index.html", error="Please enter a URL.", result=None)
    feats, names = extract_features(url)
    # Ensure feature order matches trained columns
    X = pd.DataFrame([feats], columns=names)
    # reorder or add missing columns
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_names]
    prob = model.predict_proba(X)[0][1]  # probability of phishing class
    pred = int(prob >= 0.5)
    verdict = "Phishing (likely)" if pred == 1 else "Legitimate (likely)"
    explanation = {
        'probability_phishing': float(prob),
        'features': dict(zip(feature_names, X.iloc[0].tolist()))
    }
    return render_template("index.html", result={'verdict': verdict, 'prob': f"{prob:.3f}", 'explanation': explanation, 'url': url})

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    if not data or 'url' not in data:
        return jsonify({"error": "Missing 'url' in JSON body."}), 400
    url = data['url']
    feats, names = extract_features(url)
    X = pd.DataFrame([feats], columns=names)
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_names]
    prob = model.predict_proba(X)[0][1]
    pred = int(prob >= 0.5)
    verdict = "phishing" if pred == 1 else "legitimate"
    return jsonify({
        "url": url,
        "verdict": verdict,
        "probability_phishing": float(prob),
        "features": dict(zip(feature_names, X.iloc[0].tolist()))
    })

# simple health
@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
