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

⚙️ Installation and Setup
1️⃣ Clone the repository
git clone https://github.com/sairajjapu/phishing_url_detector.git
cd phishing_url_detector

2️⃣ Create a virtual environment
python -m venv venv


Activate it:

On Windows: venv\Scripts\activate

On macOS/Linux: source venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the Flask app
python app.py


By default, the app will run on
👉 http://127.0.0.1:5000

🧩 Example Usage

Open the web app in your browser.

Enter a URL, e.g.:

http://paypal-login-verification.com/


Click Check URL.

The app will return a message such as:

⚠️ This is a Phishing URL!


or

✅ This URL appears to be Legitimate.

📦 requirements.txt (sample)
Flask==3.0.2
numpy==1.26.0
pandas==2.2.0
scikit-learn==1.5.0
joblib==1.4.0