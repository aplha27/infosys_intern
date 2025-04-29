import os
import pandas as pd
import speech_recognition as sr
import time
import pyttsx3
from together import Together
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ============================
# 1. Set API Key and Initialize Client
# ============================
os.environ["TOGETHER_API_KEY"] = "56798c1b2f78b7d4b78aea84c6a3781a32c0a620620ba3115166de8f3a7e5037"
client = Together()

# ============================
# 2. Load and Process Excel Data
# ============================
file_path = "candidate_resume.xlsx"
df = pd.read_excel(file_path)
job_description = df.loc[0, "Job Description"]
resume = df.loc[0, "Resume"]

# ============================
# 3. Initialize Speech Modules
# ============================
engine = pyttsx3.init()
recognizer = sr.Recognizer()
qa_pairs = []

# ============================
# 4. Conduct AI Interview
# ============================
for i in range(10):
    prompt = f"Based on the following job description and resume, generate a relevant interview question (generate only question without any description of question):\n\nJob Description: {job_description}\n\nResume: {resume}\n\nPrevious Q&A: {qa_pairs}\n\nNext Interview Question:"
    try:
        response = client.chat.completions.create(model="meta-llama/Llama-Vision-Free", messages=[{"role": "user", "content": prompt}])
        question = response.choices[0].message.content.strip()
    except Exception as e:
        error_message = str(e)
        if "credit_limit" in error_message or "402" in error_message:
            question = "Error: Credit limit exceeded. Please upgrade your plan or add credit."
        else:
            raise
    
    print(f"AI: {question}")
    engine.say(question)
    engine.runAndWait()
    
    with sr.Microphone() as source:
        print("Candidate, please answer:")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    
    try:
        answer = recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        answer = "[Unrecognized speech]"
    time.sleep(3)
    print(f"Candidate's Answer: {answer}")
    qa_pairs.append({"Question": question, "Answer": answer})

# Save Q&A pairs to Excel
output_df = pd.DataFrame(qa_pairs)
output_file = "interview_transcript.xlsx"
output_df.to_excel(output_file, index=False)
print(f"Interview transcript saved to {output_file}")

# ============================
# 5. Merge Resume and Transcript Data
# ============================
transcript_file = "interview_transcript.xlsx"
df_transcript = pd.read_excel(transcript_file)
df_transcript["Name"] = df["Name"].iloc[:len(df_transcript)]
df_transcript["Transcript"] = df_transcript.apply(lambda row: f"{row['Question']}\n{row['Answer']}", axis=1)
df_transcript = df_transcript.groupby("Name")["Transcript"].apply(lambda x: "\n".join(x)).reset_index()
df_transcript.to_excel(transcript_file, index=False)
print("interview_transcript.xlsx updated successfully!")

# ============================
# 6. Process Data for Model
# ============================
model_file = "xgb_model.pkl"
df_merged = pd.merge(df, df_transcript, on="Name", how="left").fillna("")

def clean_text(text):
    return re.sub(r'[^a-z0-9\s]', '', text.lower()) if pd.notnull(text) else ""

df_merged["Cleaned_Resume"] = df_merged["Resume"].apply(clean_text)
df_merged["Cleaned_Transcript"] = df_merged["Transcript"].apply(clean_text)
df_merged["Cleaned_Job_Description"] = df_merged["Job Description"].apply(clean_text)

# Feature extraction functions
def count_words(text):
    return len(text.split()) if pd.notnull(text) else 0

def avg_word_length(text):
    words = text.split() if pd.notnull(text) else []
    return sum(len(word) for word in words) / len(words) if words else 0

def unique_word_ratio(text):
    words = text.split() if pd.notnull(text) else []
    return len(set(words)) / len(words) if words else 0

def keyword_count(text, keywords):
    words = text.split() if pd.notnull(text) else []
    return sum(1 for word in words if word.lower() in keywords)

def keyword_overlap(text1, text2):
    if pd.notnull(text1) and pd.notnull(text2):
        words1 = set(text1.split())
        words2 = set(text2.split())
        return len(words1 & words2)
    return 0



technical_keywords = {'python', 'java', 'sql', 'machine learning', 'cloud', 'design', 'analysis', 'management'}
positive_keywords = {'excellent', 'success', 'outstanding', 'achievement', 'skilled'}
negative_keywords = {'poor', 'inadequate', 'lacking', 'failure', 'weak'}

# Apply feature extraction
df_merged['resume_positive_keyword_count'] = df_merged['Cleaned_Resume'].apply(lambda x: keyword_count(x, positive_keywords))
df_merged['resume_negative_keyword_count'] = df_merged['Cleaned_Resume'].apply(lambda x: keyword_count(x, negative_keywords))
df_merged['resume_char_count'] = df_merged['Cleaned_Resume'].apply(len)
df_merged['resume_job_keyword_overlap'] = df_merged.apply(lambda row: keyword_overlap(row['Cleaned_Resume'], row['Cleaned_Job_Description']), axis=1)

df_merged['transcript_positive_keyword_count'] = df_merged['Cleaned_Transcript'].apply(lambda x: keyword_count(x, positive_keywords))
df_merged['transcript_char_count'] = df_merged['Cleaned_Transcript'].apply(len)
df_merged['transcript_avg_word_length'] = df_merged['Cleaned_Transcript'].apply(avg_word_length)
df_merged['transcript_unique_word_ratio'] = df_merged['Cleaned_Transcript'].apply(unique_word_ratio)
df_merged['transcript_job_keyword_overlap'] = df_merged.apply(lambda row: keyword_overlap(row['Cleaned_Transcript'], row['Cleaned_Job_Description']), axis=1)

# ============================
# 7. Model Prediction
# ============================
xgb_model = joblib.load(model_file)
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
df_merged['Selection_Prediction'] = xgb_model.predict(df_merged[features])
df_merged.to_excel("processed_results.xlsx", index=False)
print("Processed data with predictions saved to processed_results.xlsx")

# ============================
# 8. Selection Status and Email Notification
# ============================
df_results = pd.read_excel("processed_results.xlsx")
df_results["Selection_Status"] = df_results["Selection_Prediction"].apply(lambda x: "Selected" if x >= 0.5 else "Not Selected")
print(df_results[["Name", "Selection_Status"]])

def send_email(name, selection_status):
    from_email = "2713alpha8631@gmail.com"
    from_password = "pldf ttue xzte decz"
    to_email = "6520mhari8631@gmail.com"
    
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = f"Selection Status for {name}"
    msg.attach(MIMEText(f"Candidate {name} has been {selection_status}.", 'plain'))
    
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(from_email, from_password)
        server.sendmail(from_email, to_email, msg.as_string())

for _, row in df_results.iterrows():
    send_email(row["Name"], row["Selection_Status"])


