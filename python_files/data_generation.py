import pandas as pd
import os
from together import Together

# Configure Together API
os.environ["TOGETHER_API_KEY"] = "dfd4b5dc8c148f418ea5b2702bd8721a21ca6fabe3e8dc1c511fc7abae24c0a7"
client = Together()

# Load the candidate data from the uploaded Excel file
file_path = "\infosys\data\prediction_data.xlsx"
data = pd.read_excel(file_path)

# Function to generate interview questions using Together API
def generate_interview_questions_with_llama(job_description):
    prompt = (
        f"Based on the following job description, generate at least 10 unique and insightful interview questions: \n"
        f"{job_description}\n"
        f"Ensure the questions are relevant to the role and test both technical and behavioral aspects of the candidate."
    )

    response = client.chat.completions.create(
        model="meta-llama/Llama-Vision-Free",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()

# Iterate through candidates and generate questions
interview_data = []
for index, row in data.iterrows():
    job_description = row.get("Job Description", "No description provided")
    name = row.get("Name", "Candidate")
    role = row.get("Role", "Unknown Role")

    questions = generate_interview_questions_with_llama(job_description)

    interview_data.append({
        "Candidate Name": name,
        "Role": role,
        "Job Description": job_description,
        "Interview Questions": questions
    })

# Convert to DataFrame
interview_df = pd.DataFrame(interview_data)

# Save the generated questions to a new Excel file
output_file = "\infosys\data\generated_interview_questions.xlsx"
interview_df.to_excel(output_file, index=False)
print(f"Interview questions have been generated using Llama API and saved to {output_file}.")



