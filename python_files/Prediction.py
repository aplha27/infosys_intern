import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import re

# Load prediction data
prediction_df = pd.read_excel('prediction_data.xlsx')

# Define helper functions
def count_words(text):
    return len(text.split()) if pd.notnull(text) else 0

def count_characters(text):
    return len(text) if pd.notnull(text) else 0

def avg_word_length(text):
    words = text.split() if pd.notnull(text) else []
    return sum(len(word) for word in words) / len(words) if words else 0

def count_sentences(text):
    return len(re.split(r'[.!?]', text)) - 1 if pd.notnull(text) else 0

def count_uppercase_ratio(text):
    return sum(1 for char in text if char.isupper()) / len(text) if pd.notnull(text) and len(text) > 0 else 0

def keyword_count(text, keywords):
    words = text.split() if pd.notnull(text) else []
    return sum(1 for word in words if word.lower() in keywords)

def unique_word_ratio(text):
    words = text.split() if pd.notnull(text) else []
    return len(set(words)) / len(words) if words else 0

def keyword_overlap(text1, text2):
    if pd.notnull(text1) and pd.notnull(text2):
        words1 = set(text1.split())
        words2 = set(text2.split())
        return len(words1 & words2)
    return 0

# Predefined keyword dictionaries
technical_keywords = {'python', 'java', 'sql', 'machine learning', 'cloud', 'design', 'analysis', 'management'}
positive_keywords = {'excellent', 'success', 'outstanding', 'achievement', 'skilled'}
negative_keywords = {'poor', 'inadequate', 'lacking', 'failure', 'weak'}

# Feature extraction
prediction_df['resume_word_count'] = prediction_df['Resume'].apply(count_words)
prediction_df['resume_char_count'] = prediction_df['Resume'].apply(count_characters)
prediction_df['resume_avg_word_length'] = prediction_df['Resume'].apply(avg_word_length)
prediction_df['resume_sentence_count'] = prediction_df['Resume'].apply(count_sentences)
prediction_df['resume_uppercase_ratio'] = prediction_df['Resume'].apply(count_uppercase_ratio)
prediction_df['resume_technical_keyword_count'] = prediction_df['Resume'].apply(lambda x: keyword_count(x, technical_keywords))
prediction_df['resume_positive_keyword_count'] = prediction_df['Resume'].apply(lambda x: keyword_count(x, positive_keywords))
prediction_df['resume_negative_keyword_count'] = prediction_df['Resume'].apply(lambda x: keyword_count(x, negative_keywords))
prediction_df['resume_unique_word_ratio'] = prediction_df['Resume'].apply(unique_word_ratio)

prediction_df['transcript_word_count'] = prediction_df['Transcript'].apply(count_words)
prediction_df['transcript_char_count'] = prediction_df['Transcript'].apply(count_characters)
prediction_df['transcript_avg_word_length'] = prediction_df['Transcript'].apply(avg_word_length)
prediction_df['transcript_sentence_count'] = prediction_df['Transcript'].apply(count_sentences)
prediction_df['transcript_uppercase_ratio'] = prediction_df['Transcript'].apply(count_uppercase_ratio)
prediction_df['transcript_positive_keyword_count'] = prediction_df['Transcript'].apply(lambda x: keyword_count(x, positive_keywords))
prediction_df['transcript_negative_keyword_count'] = prediction_df['Transcript'].apply(lambda x: keyword_count(x, negative_keywords))
prediction_df['transcript_unique_word_ratio'] = prediction_df['Transcript'].apply(unique_word_ratio)

prediction_df['resume_job_keyword_overlap'] = prediction_df.apply(lambda row: keyword_overlap(row['Resume'], row['Job Description']), axis=1)
prediction_df['transcript_job_keyword_overlap'] = prediction_df.apply(lambda row: keyword_overlap(row['Transcript'], row['Job Description']), axis=1)

# Decision reason encoding
prediction_df['decision_reason_encoded'] = prediction_df['Reason for decision'].astype('category').cat.codes


# Create a fresh TF-IDF vectorizer and fit it to the data
loaded_vectorizer = TfidfVectorizer()
loaded_vectorizer.fit(prediction_df['Job Description'].fillna(''))  # Fit on job descriptions

# Transform text data
new_job_desc_vectors = loaded_vectorizer.transform(prediction_df['Job Description'].fillna(''))
new_resume_vectors = loaded_vectorizer.transform(prediction_df['Resume'].fillna(''))
new_transcript_vectors = loaded_vectorizer.transform(prediction_df['Transcript'].fillna(''))

# Cosine similarities
prediction_df['resume_job_similarity'] = [
    cosine_similarity(new_resume_vectors[i], new_job_desc_vectors[i])[0][0] for i in range(len(prediction_df))
]
prediction_df['transcript_job_similarity'] = [
    cosine_similarity(new_transcript_vectors[i], new_job_desc_vectors[i])[0][0] for i in range(len(prediction_df))
]

# Load XGBoost model
with open('xgb_model.pkl', 'rb') as model_file:
    xgb_model = pickle.load(model_file)

# Feature subset for prediction
features = [
    'transcript_positive_keyword_count',
    'resume_positive_keyword_count',
    'transcript_avg_word_length',
    'decision_reason_encoded',
    'transcript_char_count',
    'transcript_job_keyword_overlap',
    'resume_negative_keyword_count',
    'resume_job_keyword_overlap',
    'resume_char_count',
    'transcript_unique_word_ratio'
]

# Make predictions
X = prediction_df[features]
prediction_df['Selection_Status'] = xgb_model.predict(X)
prediction_df['Selection_Status'] = prediction_df['Selection_Status'].apply(lambda x: 'Selected' if x == 1 else 'Not Selected')

# # Check the columns in prediction_df
# print(prediction_df.columns.tolist())  # This will help you see what columns are available

# Save the results to Excel
output_file = 'prediction_results.xlsx'
# Ensure 'Name' exists before trying to save
if 'Name' in prediction_df.columns and 'Selection_Status' in prediction_df.columns:
    prediction_df[['Name', 'Selection_Status']].to_excel(output_file, index=False)
else:
    print("Error: One or more required columns are missing from the DataFrame.")

# Email the results
def send_email(file_path):
    from_email = "2713alpha8631@gmail.com"
    from_password = "pldf ttue xzte decz"  # Update this line with your app password
    to_email = "6520mhari8631@gmail.com"

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = "Prediction Results"

    body = "Please find the attached prediction results."
    msg.attach(MIMEText(body, 'plain'))

    with open(file_path, "rb") as attachment:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f"attachment; filename={file_path}")
        msg.attach(part)

    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(from_email, from_password)
        server.sendmail(from_email, to_email, msg.as_string())

send_email(output_file)

