import streamlit as st
from rag_pipeline import rag_system
from skill_extractor import extract_skills_from_jd

st.set_page_config(page_title="AI Interview Generator", layout="centered")

st.title("AI Interview Question Generator")

jd = st.text_area("Paste Job Description")

difficulty = st.selectbox(
    "Select Difficulty",
    ["Easy", "Medium", "Hard"]
)

if st.button("Generate Questions"):
    if jd.strip() == "":
        st.warning("Please paste a job description.")
    else:
        with st.spinner("Analyzing job description..."):
            skills = extract_skills_from_jd(jd)

        st.subheader("Detected Skills")
        st.write(skills)

        st.subheader("Generated Questions")

        for skill in skills:
            st.markdown(f"### {skill}")
            result = rag_system(
                query=skill,
                skill=None,
                difficulty=difficulty
            )
            st.write(result)