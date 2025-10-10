🛡️ Phishing URL Detection

This is a simple Flask web application that detects whether a given URL is phishing or legitimate using a Machine Learning model trained on URL features.

🚀 Features

Detects phishing URLs in real-time

Built using Flask (Python)

Uses a trained Logistic Regression model (joblib file)

Simple and lightweight web interface

🧠 How It Works

The user enters a URL.

The app extracts important features from the URL.

The trained ML model predicts if it’s phishing or legitimate.

The result is shown on the web page.

⚙️ Setup Instructions
1. Clone the project
git clone https://github.com/your-username/phishing-url-detector.git
cd phishing-url-detector

2. Install required packages
pip install -r requirements.txt

3. Run the app
python app.py


Then open your browser and go to
👉 http://127.0.0.1:5000

📦 Requirements
Flask
scikit-learn
pandas
numpy
joblib

🧩 File Structure
phishing-url-detector/
│
├── app.py                  # Flask main app
├── model.pkl               # Trained ML model
├── templates/              # HTML files
├── static/                 # CSS/JS (optional)
└── requirements.txt

📦 requirements.txt (sample)
Flask==3.0.2
numpy==1.26.0
pandas==2.2.0
scikit-learn==1.5.0
joblib==1.4.0