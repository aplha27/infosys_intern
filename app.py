from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from screener import screen_resume
import pandas as pd
#import speech_recognition as sr
import pyttsx3
from together import Together
import os
import json
import joblib
import re
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Secure this in production

# Initialize AI Modules
engine = pyttsx3.init()
#recognizer = sr.Recognizer()

# Together AI API Key
os.environ["TOGETHER_API_KEY"] = "56798c1b2f78b7d4b78aea84c6a3781a32c0a620620ba3115166de8f3a7e5037"
client = Together()

# Load AI Model (if available)
model_file = "xgb_model.pkl"
xgb_model = joblib.load(model_file) if os.path.exists(model_file) else None

def clean_text(text):
    return re.sub(r'[^a-z0-9\s]', '', text.lower()) if pd.notnull(text) else ""

def send_email(name, selection_status):
    from_email = "2713alpha8631@gmail.com"
    from_password = "pldf ttue xzte decz"
    to_email = "6520mhari8631@gmail.com"
    
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = f"Selection Status for {name}"
    msg.attach(MIMEText(f"Candidate {name} has been {selection_status}.", 'plain'))
    
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(from_email, from_password)
            server.sendmail(from_email, to_email, msg.as_string())
    except Exception as e:
        print(f"Error sending email to {to_email}: {e}")
        raise

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

        try:
            response = client.chat.completions.create(
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                messages=[{"role": "user", "content": prompt}]
            )
            question = response.choices[0].message.content.strip() if response.choices else None
        except Exception as e:
            error_message = str(e)
            if "credit_limit" in error_message or "402" in error_message:
                return jsonify({'status': 'error', 'message': 'Credit limit exceeded. Please upgrade your plan or add credit.'})
            else:
                raise

        if not question:
            return jsonify({'status': 'error', 'message': 'Failed to generate interview question. Try again.'})

        # Store question in session
        session['current_question'] = question

        # Speak out the question
        # engine.say(question)
        # engine.runAndWait()

        return jsonify({'status': 'success', 'question': question})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/next_question', methods=['POST'])
def next_question():
    return jsonify({'status': 'error', 'message': 'Use /submit_answer endpoint to submit answers and get next question.'})

@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    try:
        data = request.get_json()
        answer = data.get('answer', '').strip()
        if not answer:
            return jsonify({'status': 'error', 'message': 'Answer is required.'})

        qa_pairs = json.loads(session.get('qa_pairs', '[]'))
        qa_pairs.append({"Question": session.get('current_question', ''), "Answer": answer})
        session['qa_pairs'] = json.dumps(qa_pairs)

        if len(qa_pairs) < 5:
            prompt = f"Based on the following job description and resume, generate another interview question:\n\nJob: {session.get('job_description', '')}\nResume: {session.get('resume', '')}\nPrevious Q&A: {qa_pairs}\nNext Question:"

            try:
                response = client.chat.completions.create(
                    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                    messages=[{"role": "user", "content": prompt}]
                )
                question = response.choices[0].message.content.strip() if response.choices else None
            except Exception as e:
                error_message = str(e)
                if "credit_limit" in error_message or "402" in error_message:
                    return jsonify({'status': 'error', 'message': 'Credit limit exceeded. Please upgrade your plan or add credit.'})
                else:
                    raise

            if not question:
                return jsonify({'status': 'error', 'message': 'Failed to generate next question. Try again.'})

            session['current_question'] = question
            return jsonify({
                'status': 'success',
                'question': question,
                'progress': (len(qa_pairs) / 5) * 100
            })
        else:
            return jsonify({'complete': True, 'message': 'Interview complete!'})

            # After interview complete, process results and send emails
            try:
                # Prepare data for prediction and email
                # For simplicity, save transcript and resume to temporary Excel files
                import tempfile
                import os as os_module

                temp_dir = tempfile.gettempdir()
                transcript_file = os_module.path.join(temp_dir, "interview_transcript.xlsx")
                resume_file = os_module.path.join(temp_dir, "candidate_resume.xlsx")

                # Create DataFrame from session Q&A pairs
                import pandas as pd
                qa_pairs = json.loads(session.get('qa_pairs', '[]'))
                df_transcript = pd.DataFrame(qa_pairs)
                df_transcript.rename(columns={"Question": "Question", "Answer": "Answer"}, inplace=True)
                # Save transcript to Excel
                df_transcript.to_excel(transcript_file, index=False)

                # For resume data, try to get from session or fallback
                resume_text = session.get('resume', '')
                job_description = session.get('job_description', '')
                # Create dummy resume DataFrame
                df_resume = pd.DataFrame([{
                    "Name": "Candidate",
                    "Resume": resume_text,
                    "Job Description": job_description
                }])
                df_resume.to_excel(resume_file, index=False)

                # Load and process data similar to interview.py
                df = pd.read_excel(resume_file)
                df_transcript = pd.read_excel(transcript_file)
                df_transcript["Name"] = df["Name"].iloc[:len(df_transcript)]
                df_transcript["Transcript"] = df_transcript.apply(lambda row: f"{row['Question']}\n{row['Answer']}", axis=1)
                df_transcript = df_transcript.groupby("Name")["Transcript"].apply(lambda x: "\n".join(x)).reset_index()

                df_merged = pd.merge(df, df_transcript, on="Name", how="left").fillna("")

                df_merged["Cleaned_Resume"] = df_merged["Resume"].apply(clean_text)
                df_merged["Cleaned_Transcript"] = df_merged["Transcript"].apply(clean_text)
                df_merged["Cleaned_Job_Description"] = df_merged["Job Description"].apply(clean_text)

                positive_keywords = {'excellent', 'success', 'outstanding', 'achievement', 'skilled'}
                negative_keywords = {'poor', 'inadequate', 'lacking', 'failure', 'weak'}

                def keyword_count(text, keywords):
                    words = text.split() if pd.notnull(text) else []
                    return sum(1 for word in words if word.lower() in keywords)

                def keyword_overlap(text1, text2):
                    if pd.notnull(text1) and pd.notnull(text2):
                        words1 = set(text1.split())
                        words2 = set(text2.split())
                        return len(words1 & words2)
                    return 0

                df_merged['resume_positive_keyword_count'] = df_merged['Cleaned_Resume'].apply(lambda x: keyword_count(x, positive_keywords))
                df_merged['resume_negative_keyword_count'] = df_merged['Cleaned_Resume'].apply(lambda x: keyword_count(x, negative_keywords))
                df_merged['resume_char_count'] = df_merged['Cleaned_Resume'].apply(len)
                df_merged['resume_job_keyword_overlap'] = df_merged.apply(lambda row: keyword_overlap(row['Cleaned_Resume'], row['Cleaned_Job_Description']), axis=1)

                df_merged['transcript_positive_keyword_count'] = df_merged['Cleaned_Transcript'].apply(lambda x: keyword_count(x, positive_keywords))
                df_merged['transcript_char_count'] = df_merged['Cleaned_Transcript'].apply(len)
                df_merged['transcript_avg_word_length'] = df_merged['Cleaned_Transcript'].apply(lambda x: sum(len(word) for word in x.split()) / len(x.split()) if x.split() else 0)
                df_merged['transcript_unique_word_ratio'] = df_merged['Cleaned_Transcript'].apply(lambda x: len(set(x.split())) / len(x.split()) if x.split() else 0)
                df_merged['transcript_job_keyword_overlap'] = df_merged.apply(lambda row: keyword_overlap(row['Cleaned_Transcript'], row['Cleaned_Job_Description']), axis=1)

                features = [
                    'transcript_positive_keyword_count',
                    'resume_positive_keyword_count',
                    'transcript_avg_word_length',
                    'transcript_char_count',
                    'transcript_job_keyword_overlap',
                    'resume_negative_keyword_count',
                    'resume_job_keyword_overlap',
                    'resume_char_count',
                    'transcript_unique_word_ratio'
                ]

                predictions = xgb_model.predict(df_merged[features])
                df_merged['Selection_Prediction'] = predictions
                df_merged.to_excel(os_module.path.join(temp_dir, "processed_results.xlsx"), index=False)

                df_results = df_merged.copy()
                df_results["Selection_Status"] = df_results["Selection_Prediction"].apply(lambda x: "Selected" if x >= 0.5 else "Not Selected")

                for _, row in df_results.iterrows():
                    send_email(row["Name"], row["Selection_Status"])

            except Exception as email_error:
                print(f"Error sending emails: {email_error}")

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

