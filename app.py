import streamlit as st
import PyPDF2
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Smart Resume Analyzer")
st.title("AI Smart Resume Analyzer")

def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    text=""
    for page in reader.pages:
        page_text=page.extract_text()
        if page_text:
            text+=page_text
    return text.lower()

skills_db=["python","machine learning","data science","sql","deep learning","nlp","java","cloud","aws","data analysis","pandas","tensorflow"]

uploaded_file=st.file_uploader("Upload Resume (PDF)",type=["pdf"])
job_description=st.text_area("Paste Job Description")

if uploaded_file:
    resume_text=extract_text(uploaded_file)
    st.subheader("Resume Preview")
    st.text_area("Extracted Text",resume_text,height=250)
    found=[s for s in skills_db if s in resume_text]
    st.subheader("Skills Found")
    if found:
        for s in found:
            st.write("•",s)
    else:
        st.write("No skills detected")

    score=0
    if job_description:
        vec=TfidfVectorizer()
        X=vec.fit_transform([resume_text,job_description.lower()])
        score=round(cosine_similarity(X[0],X[1])[0][0]*100,2)
        st.subheader("Resume Match Score")
        st.progress(score/100)
        st.write(f"{score}%")

    st.subheader("Resume Dashboard")
    fig,ax=plt.subplots()
    ax.bar(["Skills","Experience","Education","Formatting"],[80,70,75,85])
    st.pyplot(fig)

    st.subheader("Suggestions")
    if len(found)<5:
        st.write("• Add more technical skills")
    if "project" not in resume_text:
        st.write("• Mention project experience")
    if "internship" not in resume_text:
        st.write("• Include internship details")
