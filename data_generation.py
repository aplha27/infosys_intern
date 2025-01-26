# #import random
# #import pandas as pd
# # from together import Together
# import os
# import time
#import pandas as pd  # Ensure pandas is imported

# # Configure Together API
# os.environ["TOGETHER_API_KEY"] = "dfd4b5dc8c148f418ea5b2702bd8721a21ca6fabe3e8dc1c511fc7abae24c0a7"  # Replace with your actual API key
# client = Together()

# # Candidate Names
# names = [
#     "Aarav", "Sophia", "Neha", "Ethan", "Kunal", "Amara", "Sonia", "Liam", "Rajiv", "Emma",
#     "Meera", "Noah", "Arjun", "Isabella", "Diya", "Lucas", "Ananya", "Maya", "Ishan", "Oliver"
# ]

# # Job Roles and Skills
# roles_skills = {
#     "Data Scientist": [
#         "Statistical analysis", "Machine learning", "Data visualization",
#         "Python/R programming", "Big data tools", "SQL", "Feature engineering",
#         "Deep learning frameworks", "Cloud platforms", "Communication"
#     ],
#     "Software Engineer": [
#         "Python", "Java", "C++", "Data structures", "System design", "Git/GitHub",
#         "API development", "Agile methodologies", "Linux", "Cloud services"
#     ],
#     "Data Engineer": [
#         "ETL processes", "Big data frameworks", "Cloud data services", "SQL",
#         "Programming (Python, Scala)", "Data modeling", "Workflow orchestration tools",
#         "Real-time data streaming", "Performance optimization", "API integration"
#     ],
#     "UI Designer": [
#         "Wireframing", "Prototyping", "Typography", "Responsive design",
#         "HTML, CSS", "Graphic design tools", "Design systems", "Usability testing",
#         "Accessibility standards", "Information architecture"
#     ],
#     "Data Analyst": [
#         "Data cleaning", "Statistical analysis", "Data visualization", "SQL",
#         "Python/R", "A/B testing", "Presentation skills", "Predictive analytics",
#         "Dashboards creation", "Problem-solving"
#     ],
#     "Product Manager": [
#         "Product lifecycle management", "Market research", "Agile methodologies",
#         "Stakeholder management", "Business strategy", "User research",
#         "Wireframing tools", "KPI monitoring", "Risk management",
#         "Collaboration with teams"
#     ]
# }

# # Experience Levels and Work Environments
# experience_levels = ["Entry-level", "Mid-level", "Senior-level", "Lead", "Director"]
# work_environments = ["Remote", "Hybrid", "In-office"]

# # Randomized Result Generator
# def generate_result():
#     return random.choice(["selected", "rejected"])

# # Generate Profile
# def generate_profile(name, role, skills, result, experience, work_env, rating):
#     return {
#         "Name": name,
#         "Role": role,
#         "Experience Level": experience,
#         "Preferred Work Environment": work_env,
#         "Skills": skills,
#         "Rating": f"{rating} / 5 stars",
#     }

# # Generate Reason
# def generate_reason(result, skills, role):
#     reason = (
#         f"Candidate is {result} due to their expertise in {', '.join(skills[:3])} and proficiency in {skills[3]}. "
#         if result == "selected" else
#         f"Candidate was not selected as their experience did not align with the expectations for {role}."
#     )
#     response = client.chat.completions.create(
#         model="meta-llama/Llama-Vision-Free",
#         messages=[{"role": "user", "content": reason}]
#     )
#     return response.choices[0].message.content.strip()

# # Generate Interview Transcript Using Together API
# def generate_transcript(name, role, result, skills, experience, work_env):
#     prompt = (
#         f"Create an interview conversation between the interviewer and {name}, "
#         f"who applied for the {role} position at an {experience} level. Include questions about their "
#         f"skills ({', '.join(skills)}) and their suitability for a {work_env} work environment. "
#         f"The candidate is {result}. "
#         f"donot produce discription of the generated data"
#     )
#     response = client.chat.completions.create(
#         model="meta-llama/Llama-Vision-Free",
#         messages=[{"role": "user", "content": prompt}]
#     )
#     return response.choices[0].message.content.strip()

# # Generate Random Skills
# def get_random_skills(role):
#     skills = roles_skills[role]
#     num_skills = min(len(skills), 7)
#     return random.sample(skills, k=num_skills)

# # Generate Random Rating
# def generate_rating():
#     return round(random.uniform(3.0, 5.0), 1)  # Ratings between 3.0 and 5.0

# # Main Automation Loop with 5-Second Delay
# data = []
# for candidate_id in range(1, 11):
#     name = random.choice(names)
#     role = random.choice(list(roles_skills.keys()))
#     skills = get_random_skills(role)
#     result = generate_result()
#     experience = random.choice(experience_levels)
#     work_env = random.choice(work_environments)
#     rating = generate_rating()

#     # Generate profile, reason, and transcript
#     profile = generate_profile(name, role, skills, result, experience, work_env, rating)
#     reason = generate_reason(result, skills, role)
#     transcript = generate_transcript(name, role, result, skills, experience, work_env)

#     # Append data
#     data.append({
#         #"Candidate ID": candidate_id,
#         "Name": name,
#         "Role": role,
#         #"Experience Level": experience,
#         #"Preferred Work Environment": work_env,
#         "Transcript": transcript,
#         "Profile": profile,
#         #"Rating": rating,
#         "Result": result,
#         "Reason": reason
#     })

#     # Delay before the next iteration
#     time.sleep(5)

# # Convert to DataFrame
# df = pd.DataFrame(data)
# print(df.head())


# # Save to Excel
# #df.to_excel("candidate_results.xlsx", index=False)
# #print("Data generation complete. Results saved to candidate_results.xlsx.")

# Combine all Excel sheets in the dataset folder
import os
import pandas as pd

# Define the folder path containing the Excel files
dataset_folder = '\infosys\datset'  # Replace with your folder path

# Define the columns to focus on
columns_to_include = ["ID", "Name", "Role", "Transcript", "Resume", "decision", "Reason for decision", "Job Description"]

# Initialize an empty DataFrame to store concatenated data
concatenated_data = pd.DataFrame()

# Process each dataset file explicitly
for i in range(1, 10):  # Loop from 1 to 9
    file_name = f'dataset{i}.xlsx'
    file_path = os.path.join(dataset_folder, file_name)
    
    # Check if the file exists before processing
    if os.path.exists(file_path):
        try:
            # Read the Excel file
            data = pd.read_excel(file_path)
            
            # Filter the data to include only the specified columns
            filtered_data = data[columns_to_include]
            
            # Concatenate the filtered data to the main DataFrame
            concatenated_data = pd.concat([concatenated_data, filtered_data], ignore_index=True)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    else:
        print(f"File {file_name} not found in the folder.")

# Save the concatenated data to a new Excel file
output_file = 'combined_data.xlsx'
concatenated_data.to_excel(output_file, index=False)
print(f"Concatenated data saved to {output_file}")



