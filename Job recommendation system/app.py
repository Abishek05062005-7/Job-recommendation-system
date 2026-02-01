import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Dataset
jobs_data = {
    "Job_Title": [
        "Data Analyst", "Data Analyst", "ML Engineer", "ML Engineer",
        "Data Scientist", "Web Developer", "Backend Developer",
        "Business Analyst", "AI Engineer", "Python Developer"
    ],
    "Company": [
        "Amazon", "Flipkart", "Google", "Microsoft",
        "Infosys", "TCS", "Zoho",
        "Accenture", "IBM", "Wipro"
    ],
    "Skills": [
        "python sql excel powerbi data analysis",
        "sql excel tableau python statistics",
        "python machine learning deep learning tensorflow",
        "python ml algorithms data structures",
        "python sql machine learning statistics",
        "html css javascript react",
        "python django sql rest api",
        "excel sql business analysis powerbi",
        "python deep learning computer vision nlp",
        "python oops django flask sql"
    ]
}

df = pd.DataFrame(jobs_data)

# ML model
tfidf = TfidfVectorizer()
skill_matrix = tfidf.fit_transform(df["Skills"])

def recommend_jobs(user_skills):
    user_vec = tfidf.transform([user_skills])
    similarity = cosine_similarity(user_vec, skill_matrix)
    df["Match_Score"] = similarity[0]
    return df.sort_values(by="Match_Score", ascending=False).head(5)

# Streamlit UI
st.set_page_config(page_title="Job Recommendation System", layout="centered")
st.title("💼 Smart Job Recommendation System")
st.write("Enter your skills to get the best job matches")

user_input = st.text_input("Your Skills (comma or space separated)")

if st.button("Recommend Jobs"):
    if user_input.strip() == "":
        st.warning("Please enter at least one skill.")
    else:
        results = recommend_jobs(user_input)
        st.dataframe(results[["Job_Title", "Company", "Match_Score"]])
