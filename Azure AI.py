##Import Bin
import streamlit as st
import pandas as pd
from openai import AzureOpenAI
from PIL import Image
from pypdf import PdfReader
import re

st.set_page_config("Azure AI 2.0")
st.markdown("<h3 style='Text-align:center;'> Azure AI 2.0 </h1>",unsafe_allow_html=True)

##Tabs
t1, t2, t3, t4=st.tabs([ " 🪪 Document Q&A", "📄 AI Resume Analyzer" ,"🔉 AI Mock Interview", "✍️ AI Translator"])

#OpenAI
client=AzureOpenAI(
    api_key="1ifrcFeRyem85MYRaX9iYVcoBrIGSLPvDLlKuBPKMoVymC8vzKPAJQQJ99CBACYeBjFXJ3w3AAAAACOGEF85",
    azure_endpoint="https://azureabhiraj-openai.cognitiveservices.azure.com/",
    api_version="2024-12-01-preview"
)


with t1:
    st.markdown("<h3 style='Text-align:center;'> 🪪 Document Q&A </h1>",unsafe_allow_html=True)
    st.write("Upload a PDF → Ask questions about it → App answers based only on the Document...")
    file=st.file_uploader("Choose your Doucument ",type=["pdf","docx","txt"], key="Q&A")

    prompt=st.text_input("Enter your Prompt", key="AA") 


    if file is not None:
        if st.button("Analyze", key="Document_Q&A"):
            st.write("Analyzing...")


            reader=PdfReader(file)
            text=""
            for page in reader.pages:
                text=text+page.extract_text()
            text=re.sub(r'[^a-z0-9A-Z\s]','',text)
            st.write("Processing the Document...")


            response=client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role":"system","content":"You are a helpfull AI Assistant."},
                    {"role":"user","content":f"Document : \n{text}\n Question : {prompt}"}
                ]
            )

            st.divider()


            st.write(response.choices[0].message.content)

with t2:
    st.markdown("<h3 style='Text-align:center;'> 📄 AI Resume Analyzer </h1>",unsafe_allow_html=True)
    st.write("User uploads resume → AI analyzes → returns → Skill gaps, Suggestions to improve, Job Match Score...")

    file=st.file_uploader("Upload your Resume (PDF)", type=["pdf"], key="Resume")
    role=st.text_input("Enter your Targeted Role/Job ", key="dfcdf")
    job_description=st.text_area("Enter the Job Requirements or Job Description (Copy-Paste)", key="effdsv")
    
    if st.button("Analyze"):
        st.write("Analyzing...")
        reader=PdfReader(file)
        text=""
        for page in reader.pages:
            text=text+page.extract_text()
        text=re.sub(r'[^a-z0-9A-Z\s]','',text)

        response=client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional resume reviewer."},
                {"role": "user", "content": f"Analyze this resume for Role : \n{role}\n Resume : {text}\n Job Description : {job_description}, also give the job match score and percentage, skill gaps, suggestions to improve."}
            ]
        )

        st.divider()

        st.write(response.choices[0].message.content)


with t3:
    st.markdown("<h3 style='Text-align:center;'> 🔉 AI Mock Interviewer </h1>",unsafe_allow_html=True)
    st.write("User tells the Job Description & Resume → AI Analyzes → Gives Interview Level Q's →  → User Gives the Answer → AI Gives Feedback → Suggestions to Improve")
    file=st.file_uploader("Upload your Resume (PDF)", type=["pdf"], key="Inter")
    role=st.text_input("Enter your Targeted Role/Job ", key="huloo")
    company=st.text_input("Enter the Company Name (e.g. Microsoft)", key="comap")
    job_description=st.text_area("Enter the Job Requirements or Job Description (Copy-Paste)", key="jobbbbb")

    if "question" not in st.session_state:
        st.session_state.question=None

    if "resume_text" not in st.session_state:
        st.session_state.resume_text=None

    if st.button("Start Interview"):
        st.write("Analyzing...")
        reader=PdfReader(file)
        text=""
        for page in reader.pages:
            text=text+page.extract_text()
        text=re.sub(r'[^a-z0-9A-Z\s]','',text)
        st.session_state.resume_text=text

    def generate_question(job_description,company,role,resume):
        response=client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role":"system","content":f"You are a professional interviewer for this company : {company}."},
                {"role":"user","content":f"Analyze these for the Interview, Role : \n{role}\n Resume : {resume}\n Job Description : {job_description}\n Company : {company},also ask the user only one Real- Life Interview Question for this company, ask Technical Questions, only the question, nothing else."}
            ]
        )
        return response.choices[0].message.content

    def grade_answer(question,user_answer):
        prompt=f"You are a strict technical interviewer.\n Question : {question}\n Candidate answer : {user_answer}\n Evaluate the answer like this : \n score : number (0-10),\n strengths : ,\n weaknesses : ,\n improvements : ,\n conclusion : . Be critical and realistic."
        response=client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role":"system","content":"You are a strict technical interviewer."},
                {"role":"user","content":prompt}
            ]
        )
        return response.choices[0].message.content

    if st.button("Next Question"):
        if st.session_state.resume_text is not None:
            st.session_state.question=generate_question(job_description,company,role,st.session_state.resume_text)

    if st.session_state.question:
        st.write(st.session_state.question)
        user_answer=st.text_area("Your Answer")

        if st.button("Submit Answer"):
            result=grade_answer(st.session_state.question,user_answer)
            st.write(result)

with t4:
    st.markdown("<h3 style='Text-align:center;'> ✍️ AI Translator </h1>",unsafe_allow_html=True)
    st.write("User gives the AI a text → AI Analyzes → Auto Detects → Translate it")
    text=st.text_area("Enter the Text", key="text")
    language=st.text_input("Enter the Language to Convert into", key="language")

    if st.button("Translate"):
        st.write("Analyzing...")
        response=client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"You are a professional translator"},
                {"role": "user", "content": f"Translate this text : {text}\n to {language}\n, then tell the user what language you converted into e.g. (hindi → english), only give the translated text, and tell the user what language you converted it into, nothing else."}
            ]
        )

        st.divider()

        st.write(response.choices[0].message.content)


