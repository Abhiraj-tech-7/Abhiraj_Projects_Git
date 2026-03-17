import requests

Key="8gGaH33Wu2wfpba33Fu7gnmVuEezmL6nJcwGx8Ax9cEmhP4Y1GiVJQQJ99CCACYeBjFXJ3w3AAAEACOGR3C7"
Endpoint="https://azureabhiraj-cognitiveservices-vision.cognitiveservices.azure.com/"

person_group_id="people-database"
url=f"{Endpoint}/face/v1.0/persongroups/{person_group_id}"

headers={
    "Ocp-Apim-Subscription-Key": Key
}

data={
    "name":"People DataBase"
}

response=requests.put(url, headers=headers, json=data)
print(response.status_code)