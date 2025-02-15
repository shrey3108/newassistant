from flask import Flask, request, render_template
import sqlite3
import openai  # Replace with Gemini's equivalent when available
import google.generativeai as genai



genai.configure(api_key="AIzaSyBwejNLB1HWEO7NUQAPF4oYPBWGHSAyd1k")

model = genai.GenerativeModel('gemini-1.5-flash')

# Flask app setup
app = Flask(__name__)

# Step 1: OpenAI API Key (Replace with Gemini when available)
  # Replace with your Gemini API key


# Step 2: Database function to fetch animal history
def get_animal_history(animal_id):
    conn = sqlite3.connect('animal_health.db')
    cursor = conn.cursor()

    cursor.execute('''
    SELECT Date, Symptoms, Diagnosis, Treatment 
    FROM AnimalHistory
    WHERE Animal_ID = ?
    ''', (animal_id,))

    records = cursor.fetchall()
    conn.close()

    if records:
        return records
    else:
        return None


# Step 3: Query AI Model
def query_ai_model(animal_id, current_symptoms):
    history = get_animal_history(animal_id)
    if not history:
        return f"No history found for Animal ID {animal_id}."

    history_text = "\n".join(
        [f"Date: {date}, Symptoms: {symptoms}, Diagnosis: {diagnosis}, Treatment: {treatment}"
         for date, symptoms, diagnosis, treatment in history]
    )

    prompt = f"""
    The following is the medical history of an animal (ID: {animal_id}):
    {history_text}

    Current symptoms reported: {current_symptoms}

    Based on the historical data and the current symptoms, provide:
    1. Likely diagnosis.
    2. Suggested treatment plan.
    """

    response = openai.Completion.create(
        engine="text-davinci-003",  # Replace with Gemini's equivalent
        prompt=prompt,
        max_tokens=300,
        temperature=0.7
    )
    return response.choices[0].text.strip()


# Step 4: Routes for the web app
@app.route("/")
def home():
    return render_template("index.html")  # HTML file for the home page


@app.route("/chat", methods=["POST"])
def chat():
    # Get data from the form
    animal_id = request.form.get("animal_id")
    current_symptoms = request.form.get("current_symptoms")

    # Fetch history and query AI
    history = get_animal_history(animal_id)
    if not history:
        return render_template("index.html", result=f"No history found for Animal ID {animal_id}.")

    ai_response = query_ai_model(animal_id, current_symptoms)
    return render_template("index.html", history=history, result=ai_response)


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
