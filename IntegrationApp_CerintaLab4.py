from flask import Flask, render_template, request, send_file, jsonify
from flask_cors import CORS
import spacy
import joblib

app = Flask(__name__)
CORS(app, resources={r"/evaluate": {"origins": "http://localhost:3000"}})

nlp = spacy.load("ro_core_news_sm")

classifier = joblib.load('trained_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

questions = [
    "Cât de des ai dureri de cap?",
    "Ai dificultăți în respirație?",
    "Cum te simți în legătură cu starea ta emoțională?",
    "Ai avut probleme digestive recente?",
    "Cum ți-e vederea?",
    "Cât de des faci sport?",
    "Ai avut vreo accidentare recentă?",
    "Cum te simți după mese?",
    "Ai probleme de somn?",
    "Cum ți se pare tensiunea arterială?"
]


def evaluate_gravity(symptoms):
    keywords_high = ["durere severă", "insuficiență", "pierderea cunoștinței", "hemoragie"]
    keywords_medium = ["durere moderată", "amețeală", "probleme persistente", "disconfort"]

    severity_score = 0
    for symptom in symptoms:
        if any(keyword in symptom.lower() for keyword in keywords_high):
            severity_score += 2
        elif any(keyword in symptom.lower() for keyword in keywords_medium):
            severity_score += 1

    if severity_score >= 3:
        return "Gravitate ridicată"
    elif severity_score == 2:
        return "Gravitate moderată"
    else:
        return "Gravitate scăzută"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_responses = [request.form.get(f"question_{i}") for i in range(len(questions))]
        user_responses_text = " ".join(user_responses)

        doc = nlp(user_responses_text)
        extracted_symptoms = [ent.text for ent in doc.ents if ent.label_ == "SYMPTOM"]

        symptoms_text = " ".join(extracted_symptoms) if extracted_symptoms else user_responses_text
        symptoms_vector = vectorizer.transform([symptoms_text])
        specialist_recommendation = classifier.predict(symptoms_vector)[0]

        severity = evaluate_gravity(extracted_symptoms)

        return render_template("index.html", questions=enumerate(questions),
                               recommendation=specialist_recommendation, severity=severity)

    return render_template("index.html", questions=enumerate(questions), recommendation=None, severity=None)

@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json()

    print("Received Data:", data) 

    if not data:
        return jsonify({"error": "No data provided"}), 400

    user_responses = [data.get(f"question_{i}") for i in range(len(questions))]
    user_responses_text = " ".join(user_responses)

    doc = nlp(user_responses_text)
    extracted_symptoms = [ent.text for ent in doc.ents if ent.label_ == "SYMPTOM"]

    symptoms_text = " ".join(extracted_symptoms) if extracted_symptoms else user_responses_text
    symptoms_vector = vectorizer.transform([symptoms_text])
    specialist_recommendation = classifier.predict(symptoms_vector)[0]

    severity = evaluate_gravity(extracted_symptoms)

    # Returnează rezultatul ca JSON
    return jsonify({
        "recommendation": specialist_recommendation,
        "severity": severity
    })


if __name__ == "__main__":
    app.run(debug=True)


'''
4. Perfectionare a aplicației (funcționalitate extinsă)
a. Feedback continuu pentru îmbunătățirea modelului
Adaugă un sistem de feedback pentru utilizatori, astfel încât aplicația să învețe din greșeli:

Exemple:
După predicție, utilizatorul poate confirma dacă recomandarea a fost corectă.
Datele noi pot fi stocate și utilizate pentru re-antrenarea periodică a modelului.
b. Generarea de rapoarte pentru utilizatori
Permite utilizatorilor să genereze un raport al simptomelor și recomandărilor în format PDF:

Exemplu:
from fpdf import FPDF

def generate_report(recommendation, severity, symptoms):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Raport Medical", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Recomandare: {recommendation}", ln=True)
    pdf.cell(200, 10, txt=f"Gravitate: {severity}", ln=True)
    pdf.cell(200, 10, txt=f"Simptome: {', '.join(symptoms)}", ln=True)
    pdf.output("report.pdf")
    
b. Cache pentru modelele încărcate
Modelele SpaCy și clasificadorul pot consuma multă memorie:

Încarcă-le o singură dată și folosește un sistem de cache pentru a reduce timpul de procesare.
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

c. Vectorizare mai eficientă
Înlocuiește TfidfVectorizer cu un model de sentence embeddings (de exemplu, SBERT):

sentence-transformers optimizează timpul de vectorizare și oferă reprezentări dense și eficiente:
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
symptoms_vector = embedder.encode([symptoms_text])
'''