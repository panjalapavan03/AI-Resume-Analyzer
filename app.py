import streamlit as st
import PyPDF2
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(page_title="AI Smart Resume Analyzer", page_icon="📄")

st.title("📄 AI Smart Resume Analyzer")
st.write("Upload your resume and compare it with a Job Description.")

# ---------------- PDF TEXT EXTRACTION ----------------

def extract_text(file):
    reader = PyPDF2.PdfReader(file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    return text.lower()

# ---------------- SKILL DATABASE ----------------

skills_db = [
    "python",
    "java",
    "c",
    "c++",
    "machine learning",
    "deep learning",
    "data science",
    "sql",
    "nlp",
    "tensorflow",
    "pandas",
    "numpy",
    "power bi",
    "tableau",
    "aws",
    "cloud",
    "excel",
    "html",
    "css",
    "javascript",
    "react",
    "git"
]

# ---------------- INPUT ----------------

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

job_description = st.text_area("Paste Job Description")

# ---------------- MAIN ----------------

if uploaded_file:

    resume_text = extract_text(uploaded_file)

    st.subheader("Resume Preview")

    st.text_area(
        "Extracted Resume Text",
        resume_text,
        height=250
    )

    # ---------------- SKILL EXTRACTION ----------------

    found_skills = []

    for skill in skills_db:
        if skill in resume_text:
            found_skills.append(skill)

    st.subheader("Detected Skills")

    if found_skills:
        for skill in found_skills:
            st.write("✅", skill)
    else:
        st.warning("No skills detected.")

    # ---------------- SCORE ----------------

    if job_description:

        jd = job_description.lower()

        # TF-IDF Similarity

        vectorizer = TfidfVectorizer(stop_words="english")

        vectors = vectorizer.fit_transform([resume_text, jd])

        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

        similarity_score = similarity * 100

        # ---------------- Skill Matching ----------------

        job_skills = []

        for skill in skills_db:
            if skill in jd:
                job_skills.append(skill)

        matched_skills = []

        for skill in job_skills:
            if skill in resume_text:
                matched_skills.append(skill)

        if len(job_skills) > 0:
            skill_score = (len(matched_skills) / len(job_skills)) * 100
        else:
            skill_score = 50

        # ---------------- Bonus Score ----------------

        bonus = 0

        if "project" in resume_text:
            bonus += 5

        if "internship" in resume_text:
            bonus += 5

        if "certification" in resume_text:
            bonus += 5

        if "education" in resume_text:
            bonus += 5

        # ---------------- Final Score ----------------

        score = (0.6 * skill_score) + (0.4 * similarity_score)

        score = score + bonus

        score = round(min(score, 95), 2)

        # Optional: Minimum score

        if score < 40:
            score = 40

        # ---------------- OUTPUT ----------------

        st.subheader("Resume Match Score")

        st.progress(score / 100)

        st.success(f"Overall Resume Score: {score}%")

        st.write("### Skill Match")

        st.write(f"Matched Skills: **{len(matched_skills)} / {len(job_skills)}**")

        if matched_skills:
            st.write(", ".join(matched_skills))

    # ---------------- DASHBOARD ----------------

    st.subheader("Resume Dashboard")

    skill_strength = min(len(found_skills) * 10, 100)

    experience = 80 if "internship" in resume_text else 50

    projects = 90 if "project" in resume_text else 40

    education = 90 if "education" in resume_text else 70

    categories = [
        "Skills",
        "Projects",
        "Education",
        "Experience"
    ]

    values = [
        skill_strength,
        projects,
        education,
        experience
    ]

    fig, ax = plt.subplots(figsize=(6,4))

    ax.bar(categories, values)

    ax.set_ylim(0,100)

    ax.set_ylabel("Score")

    ax.set_title("Resume Evaluation")

    st.pyplot(fig)

    # ---------------- CAREER ----------------

    st.subheader("Career Recommendations")

    if "machine learning" in found_skills:
        st.write("• Machine Learning Engineer")

    if "python" in found_skills:
        st.write("• Python Developer")

    if "data science" in found_skills:
        st.write("• Data Scientist")

    if "sql" in found_skills:
        st.write("• Data Analyst")

    # ---------------- COURSES ----------------

    st.subheader("Recommended Courses")

    if "machine learning" not in found_skills:
        st.write("• Machine Learning Fundamentals")

    if "sql" not in found_skills:
        st.write("• SQL for Beginners")

    if "aws" not in found_skills:
        st.write("• AWS Cloud Practitioner")

    if "python" not in found_skills:
        st.write("• Complete Python Bootcamp")

    # ---------------- SUGGESTIONS ----------------

    st.subheader("Resume Suggestions")

    if len(found_skills) < 5:
        st.warning("Add more technical skills.")

    if "project" not in resume_text:
        st.warning("Add project experience.")

    if "internship" not in resume_text:
        st.warning("Include internship experience.")

    if "certification" not in resume_text:
        st.warning("Mention certifications.")

    if "github" not in resume_text:
        st.warning("Add GitHub profile.")

    if "linkedin" not in resume_text:
        st.warning("Add LinkedIn profile.")
