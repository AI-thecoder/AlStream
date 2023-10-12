import streamlit as st


# Use a pipeline as a high-level helper
from transformers import pipeline

# pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

st.title("Image Captioning App 🤗")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
# Check if an image was uploaded
if uploaded_file is not None:
    # Display the selected image
    st.image(uploaded_file, caption="Selected Image", use_column_width=True)

#user_input = st.text_input("Enter some text:", "")

#response = classifier(user_input)
#result=response[0].get('label')




import requests

API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
API_TOKEN='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'
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()











from tempfile import NamedTemporaryFile





from PIL import Image

if uploaded_file is not None:
  # image = Image.open(uploaded_file)
  responce = ''
  with NamedTemporaryFile(dir='.', suffix='.jpeg') as f:
    f.write(uploaded_file.getbuffer())
    response = query(f.name)
    # st.text(f"The sentiment analysis for ` {user_input} ` is: {result}")
  st.text(response[0]["generated_text"])