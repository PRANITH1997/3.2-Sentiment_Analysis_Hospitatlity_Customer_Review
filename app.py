from flask import Flask, request, render_template
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk


nltk.download('wordnet')
nltk.download('stopwords')

app = Flask(__name__)

def preprocess_text(raw_text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    sentence = re.sub(r'[^\w\s]|[\d]', " ", raw_text)
    sentence = sentence.lower()
    tokens = sentence.split()
    clean_tokens = [t for t in tokens if t not in stop_words]
    clean_tokens = [lemmatizer.lemmatize(word) for word in clean_tokens]
    clean_tokens = " ".join(clean_tokens)
    return clean_tokens

model = joblib.load("best_models/svc.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    cleaned_text = preprocess_text(text)
    prediction = model.predict([cleaned_text])[0]
    sentiment = 'Positive' if prediction == 1 else 'Negative'

    return render_template('result.html', text=text, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)