import streamlit as st
import pandas as pd
from huggingface_hub import InferenceClient
from pypdf import PdfReader
import re
HF_TOKEN_Ex="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"
model_name1="Qwen/Qwen2.5-Coder-32B-Instruct"
st.set_page_config("Codex Ultra 2.0")
with st.sidebar:
    HF_TOKEN=st.text_input("Enter your HuggingFace Token", key='HF_TOKEN!')
    if HF_TOKEN:
        pass
    else:
        HF_TOKEN=HF_TOKEN_Ex
    model_name=st.text_input("Enter the Model you want to Use", key="Model_Name!")
    if model_name:
        pass
    else:
        model_name=model_name1
    st.divider()
    task=st.multiselect("What Task?", ["Debugging", "Code Generating", "BrainStorm"], max_selections=1, default=["Code Generating"], key="task!")
    file=st.file_uploader("Upload your Dataset", type=["pdf","docx","txt"], key="Dataset!")
    code_file=st.file_uploader("Upload your Code File", type=["pdf","docx","txt"], key="Code_File!")
    tokens=st.slider("Select number of tokens", min_value=800, max_value=2040, value=1500)


client=InferenceClient(model=model_name, token=HF_TOKEN)

st.title("🤖 Codex Ultra 2.0")

prompt=st.chat_input("Enter your Prompt", key="prompt!")
error=st.text_area("Enter your Error", key="error!")
code=st.text_area("Enter your Code", key="Code!")


temperature=0.2

if "Code Generating" in task:
    temperature=0.0
elif "Debugging" in task:
    temperature=0.3
elif "BrainStorm" in task:
    temperature=0.5

st.divider()

if code_file:
    with st.spinner("Fetching Data..."):
        reader=PdfReader(code_file)
        text1=""
        for page in reader.pages:
            text1=text1+page.extract_text()
        text1=re.sub(r'[^\x00-\x7F]+', '', text1)
else:
    text1="None"

if file:
    with st.spinner("Fetching Data..."):
        reader=PdfReader(file)
        text2=""
        for page in reader.pages:
            text2=text2+page.extract_text()
        text2=re.sub(r'[^\x00-\x7F]+', '', text2)
else:
    text2="None"

if prompt:
    st.divider()

    messages = [
        {
            "role": "system",
            "content": """
    You are a senior software engineer.

    STRICT RULES:
    - Always include Explanations for the Code!
    - Always validate syntax before responding!
    """
        },
        {
            "role": "user",
            "content": f"""
    Question:
    {prompt}

    Code:
    {code}

    Dataset:
    {text2}

    Code File:
    {text1}

    Error:
    {error}
    """
        }
    ]

    with st.spinner("Generating Answer..."):
        response=client.chat_completion(
            messages=messages,
            max_tokens=2000,
            temperature=temperature
        )

        st.write(response.choices[0].message.content)
    
