##Import Bin
import streamlit as st
from openai import AzureOpenAI
import requests
from PIL import Image
from azure.storage.blob import BlobServiceClient


st.set_page_config("Azure AI 2.0")
st.markdown("<h3 style='Text-align:center;'> 👤 People DataBase 2.0 </h1>",unsafe_allow_html=True)

##Tabs
t1, t2=st.tabs(["📊 Find People's Information", "👤 Upload People's Information"])

#OpenAI
client=AzureOpenAI(
    api_key="1ifrcFeRyem85MYRaX9iYVcoBrIGSLPvDLlKuBPKMoVymC8vzKPAJQQJ99CBACYeBjFXJ3w3AAAAACOGEF85",
    azure_endpoint="https://azureabhiraj-openai.cognitiveservices.azure.com/",
    api_version="2024-12-01-preview"
)

#Vision API
Key="8gGaH33Wu2wfpba33Fu7gnmVuEezmL6nJcwGx8Ax9cEmhP4Y1GiVJQQJ99CCACYeBjFXJ3w3AAAEACOGR3C7"
Endpoint="https://azureabhiraj-cognitiveservices-vision.cognitiveservices.azure.com/"

#Storage
connection_string="DefaultEndpointsProtocol=https;AccountName=azureabhirajdatalakegen1;AccountKey=d6+3ezBYc4HQmHZjDlocoxhfSVU0wBsvx8FQq/bmTcd7MHgFKBiMnQ9yJhaF8J0EVX7txlpPnWZX+AStuURHyw==;EndpointSuffix=core.windows.net"
blob_service=BlobServiceClient.from_connection_string(connection_string)

with t1:
    st.markdown("<h3 style='Text-align:center;'> 📊 Find People's Information </h1>",unsafe_allow_html=True)
    image_1=st.file_uploader("Upload Face Image", key="Main_Image")

    if image_1 is not None:
        image=Image.open(image_1)
        st.image(image)

        if st.button("Detect"):

            #Detect Face_ID
            detect_url=f"{Endpoint}/face/v1.0/detect"

            headers3={
                "Ocp-Apim-Subscription-Key":Key,
                "Content-Type":"application/octet-stream"
            }

            response3=requests.post(detect_url, headers=headers3, data=image_1.getvalue())

            face_id1=response3.json()[0]["faceId"]


            #Identify Person
            identify_url=f"{Endpoint}/face/v1.0/identify"

            data={
                "personGroupId":"people-database",
                "faceIds":[face_id1],
                "maxNumOfCandidatesReturned":1,
                "confidenceThreshold":0.5
            }

            result=requests.post(identify_url, headers=headers3, json=data)

            result_json=result.json()


            if result_json and len(result_json[0]["candidates"])>0:

                person_id=result_json[0]["candidates"][0]["personId"]
                confidence=result_json[0]["candidates"][0]["confidence"]


                #Get Person Name
                person_url=f"{Endpoint}/face/v1.0/persongroups/people-database/persons/{person_id}"

                person=requests.get(person_url, headers=headers3)

                person_data=person.json()

                name=person_data["name"]
                info=person_data["userData"]


                st.success("Person Found")

                st.write(f"**Name:** {name}")
                st.write(f"**Confidence:** {round(confidence*100,2)}%")

                st.markdown("### Stored Information")
                st.write(info)

            else:
                st.error("No Person Found in Database")

with t2:
    st.markdown("<h3 style='Text-align:center;'> 👤 Upload People's Information </h1>",unsafe_allow_html=True)
    container=blob_service.get_container_client("faces")

    name=st.text_input("Enter the Person's Name", key="Name_1")
    age=st.number_input(f"Enter {name}'s Age", key="Age_1", step=1)
    crime=st.text_area("Enter the Person's Crime (e.g.  1. Shoplifting - Under Section 322 or NONE)", key="crime_1")
    location=st.text_area(f"Enter {name}'s Last Known Location (e.g. Canada, BC, Surrey, near 72 Ave) ", key="location_1")
    level=st.text_area(f"Enter the {name}'s Risk Level (e.g. High or NONE)")
    notes=st.text_area(f"Enter every known Information related to {name}")
    image_file1=st.file_uploader(f"Upload {name}'s image", type=["jpg","jpeg","png"], key="Image_File1")


    if name and age and crime and location and level and notes and image_file1:
        #Upload Image to Blob
        container.upload_blob(name=f"{name}.jpg", data=image_file1.getvalue(), overwrite=True)

        person_group_id="people-database"

        ##Add Person
        url=f"{Endpoint}/face/v1.0/persongroups/{person_group_id}/persons"

        headers={
            "Ocp-Apim-Subscription-Key": Key
        }
        data={
            "name":name,
            "userData":f"""
            Age:{age}
            Crime:{crime}
            Location:{location}
            Level:{level}
            Notes:{notes}
            """
        }
        response=requests.post(url, headers=headers, json=data)

        person_id=response.json().get("personId")

        if person_id is None:
            st.error(response.json())
            st.stop()



        #Add Face to the Person
        url1=f"{Endpoint}/face/v1.0/persongroups/{person_group_id}/persons/{person_id}/persistedFaces"

        headers1={
            "Ocp-Apim-Subscription-Key":Key,
            "Content-Type":"application/octet-stream"
        }

        response1=requests.post(url1, headers=headers1, data=image_file1.getvalue())


        url2=f"{Endpoint}/face/v1.0/persongroups/{person_group_id}/train"

        headers2={
            "Ocp-Apim-Subscription-Key":Key
        }

        response2=requests.post(url2, headers=headers2)

        st.success(f"{name}'s Inforamtion Added...")
        st.success("ThankYou...")


