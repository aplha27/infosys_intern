from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from screener import screen_resume
import pandas as pd
import speech_recognition as sr
import pyttsx3
from together import Together
import os
import json
import joblib

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Secure this in production

# Initialize AI Modules
engine = pyttsx3.init()
recognizer = sr.Recognizer()

# Together AI API Key
os.environ["TOGETHER_API_KEY"] = "dfd4b5dc8c148f418ea5b2702bd8721a21ca6fabe3e8dc1c511fc7abae24c0a7"
client = Together()

# Load AI Model (if available)
model_file = "xgb_model.pkl"
xgb_model = joblib.load(model_file) if os.path.exists(model_file) else None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/screen', methods=['POST'])
def screen():
    try:
        resume_text = ""

        # Check if a file is uploaded
        if 'resume_file' in request.files:
            file = request.files['resume_file']
            if file.filename.endswith('.txt'):
                resume_text = file.read().decode('utf-8')
            elif file.filename.endswith('.xlsx'):
                df = pd.read_excel(file)
                resume_text = str(df.iloc[0][0])  # Extract first cell
            elif file.filename:  # Invalid file format
                return jsonify({'status': 'error', 'message': 'Unsupported file format'})

        # If no file, check for manually typed text
        if not resume_text:
            resume_text = request.form.get('resume_text', '')

        # Ensure required inputs exist
        job_description = request.form.get('job_description', '')
        if not resume_text or not job_description:
            return jsonify({'status': 'error', 'message': 'Resume and Job Description are required'})

        # Process screening
        result = screen_resume(job_description, resume_text)
        similarity_score = result['similarity_score']

        # Store session data
        session['resume'] = resume_text
        session['job_description'] = job_description
        session['qa_pairs'] = json.dumps([])

        # Redirect if score >= 60
        if similarity_score >= 60:
            return jsonify({
                'status': 'success',
                'qualified': True,
                'similarity_score': similarity_score,
                'match_category': result['match_category'],
                'redirect': url_for('interview')
            })

        return jsonify({
            'status': 'success',
            'qualified': False,
            'similarity_score': similarity_score,
            'match_category': result['match_category'],
            'message': f'Match Category: {result["match_category"]}'
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/interview')
def interview():
    if 'resume' not in session or 'job_description' not in session:
        return redirect('/')
    return render_template('interview.html')

@app.route('/start_interview', methods=['POST'])
def start_interview():
    try:
        resume = session.get('resume', '')
        job_description = session.get('job_description', '')
        if not resume or not job_description:
            return jsonify({'status': 'error', 'message': 'Session expired. Please rescreen your resume.'})

        # Retrieve previous Q&A (if any)
        qa_pairs = json.loads(session.get('qa_pairs', '[]'))

        # Generate an AI-powered question
        prompt = f"Based on the following job description and resume, generate an interview question:\n\nJob: {job_description}\nResume: {resume}\nPrevious Q&A: {qa_pairs}\nNext Question:"

        response = client.chat.completions.create(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract AI-generated question
        question = response.choices[0].message.content.strip() if response.choices else None

        if not question:
            return jsonify({'status': 'error', 'message': 'Failed to generate interview question. Try again.'})

        # Store question in session
        session['current_question'] = question

        # Speak out the question
        engine.say(question)
        engine.runAndWait()

        return jsonify({'status': 'success', 'question': question})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/next_question', methods=['POST'])
def next_question():
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        try:
            answer = recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            answer = "[Unrecognized speech]"
        except sr.RequestError:
            answer = "[Speech recognition service unavailable]"

        qa_pairs = json.loads(session.get('qa_pairs', '[]'))
        qa_pairs.append({"Question": session.get('current_question', ''), "Answer": answer})
        session['qa_pairs'] = json.dumps(qa_pairs)

        if len(qa_pairs) < 5:
            # Generate next question
            prompt = f"Based on the following job description and resume, generate another interview question:\n\nJob: {session.get('job_description', '')}\nResume: {session.get('resume', '')}\nPrevious Q&A: {qa_pairs}\nNext Question:"

            response = client.chat.completions.create(
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                messages=[{"role": "user", "content": prompt}]
            )

            question = response.choices[0].message.content.strip() if response.choices else None

            if not question:
                return jsonify({'status': 'error', 'message': 'Failed to generate next question. Try again.'})

            # Speak out the next question
            engine.say(question)
            engine.runAndWait()

            session['current_question'] = question
            return jsonify({
                'status': 'success',
                'question': question,
                'last_answer': answer,
                'progress': (len(qa_pairs) / 5) * 100
            })
        else:
            return jsonify({'complete': True, 'message': 'Interview complete!'})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

