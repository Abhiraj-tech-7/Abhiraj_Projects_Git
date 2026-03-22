import streamlit as st
from PIL import Image
from insightface.app import FaceAnalysis
import os
import json
import faiss
import numpy as np
from io import BytesIO

# Streamlit setup
st.set_page_config("People DataBase 2.0")
st.markdown("<h3 style='Text-align:center;'> 👤 People DataBase 2.0 </h3>", unsafe_allow_html=True)
t1,t2=st.tabs(["📊 Find People's Information","👤 Upload People's Information"])

# Paths
faces_folder="./people_database/faces"
metadata_folder="./people_database/metadata"
os.makedirs(faces_folder, exist_ok=True)
os.makedirs(metadata_folder, exist_ok=True)

# InsightFace model
@st.cache_resource
def load_model():
    app=FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640,640))
    return app

face_app=load_model()


index_path="people_index.faiss"
embedding_dim=512
if os.path.exists(index_path):
    index=faiss.read_index(index_path)
else:
    index=faiss.IndexFlatL2(embedding_dim)

ids_path="person_ids.json"
if os.path.exists(ids_path):
    with open(ids_path) as f:
        person_ids=json.load(f)
else:
    person_ids=[]


def get_embedding(image):
    img_np=np.array(image)
    faces=face_app.get(img_np)
    if len(faces)==0:
        raise ValueError("No face detected in the image.")
    embedding=np.array(faces[0].normed_embedding, dtype="float32").reshape(1,-1)
    return embedding

with t1:
    st.markdown("<h3 style='Text-align:center;'> 📊 Find People's Information </h3>", unsafe_allow_html=True)
    image_file=st.file_uploader("Upload Face Image", key="Main_Image")
    if image_file is not None:
        image=Image.open(BytesIO(image_file.getvalue())).convert("RGB")
        st.image(image)
        if st.button("Detect"):
            try:
                embedding=get_embedding(image)
                if len(person_ids)==0:
                    st.warning("No People Found in DataBase...")
                else:
                    D,I=index.search(embedding, k=1)
                    matched_person=person_ids[I[0][0]]
                    distance=D[0][0]
                    meta_file=os.path.join(metadata_folder,f"{matched_person}.json")
                    with open(meta_file) as f:
                        metadata=json.load(f)
                    st.success(f"Person Found: {metadata['name']}")
                    st.write(f"Distance Score (Lower -> Better): {distance:.4f}")
                    st.divider()
                    
                    st.title("Person's Information")

                    st.write(metadata)
            except Exception as e:
                st.error(f"No face detected: {e}")

with t2:
    st.markdown("<h3 style='Text-align:center;'> 👤 Upload People's Information </h3>", unsafe_allow_html=True)
    name=st.text_input("Enter the Person's Name")
    age=st.number_input(f"Enter {name}'s Age", step=1)
    crime=st.text_area("Enter Crime (e.g. NONE)")
    location=st.text_area("Enter Last Known Location")
    level=st.text_area("Enter Risk Level")
    notes=st.text_area("Enter Additional Notes")
    image_file1=st.file_uploader(f"Upload {name}'s image", type=["jpg","jpeg","png"])

    if name and age and crime and location and level and notes and image_file1:
        img_path=os.path.join(faces_folder,f"{name}_{len(person_ids)+1}.jpg")
        with open(img_path,"wb") as f:
            f.write(image_file1.getvalue())
        meta_path=os.path.join(metadata_folder,f"{name}.json")
        metadata_dict={"name":name,"age":age,"crime":crime,"last_known_location":location,"risk_level":level,"notes":notes}
        with open(meta_path,"w") as f:
            json.dump(metadata_dict,f)
        image=Image.open(BytesIO(image_file1.getvalue())).convert("RGB")
        try:
            embedding=get_embedding(image)
            index.add(embedding)
            person_ids.append(name)
            faiss.write_index(index,index_path)

            with open(ids_path,"w") as f:
                json.dump(person_ids,f)
            st.success(f"{name}'s Information Added Successfully")
        except Exception as e:
            st.error(f"No face detected in uploaded image: {e}")