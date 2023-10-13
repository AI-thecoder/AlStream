import streamlit as st
from transformers import pipeline
import requests
from tempfile import NamedTemporaryFile
from PIL import Image


home_tab, module1_tab, module2_tab = st.tabs(["home", "Image Captioning", "Description to Image"])
with module1_tab:
    # pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
    # st.title("Image Captioning App")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
    # Check if an image was uploaded
    if uploaded_file is not None:
        # Display the selected image
        st.image(uploaded_file, caption="Selected Image", use_column_width=True)

    #user_input = st.text_input("Enter some text:", "")

    #response = classifier(user_input)
    #result=response[0].get('label')

    API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
    API_TOKEN='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'
    headers = {"Authorization": f"Bearer {API_TOKEN}"}

    def query(filename):
        with open(filename, "rb") as f:
            data = f.read()
        response = requests.post(API_URL, headers=headers, data=data)
        return response.json()


    if uploaded_file is not None:
        # image = Image.open(uploaded_file)
        responce = ''
        with NamedTemporaryFile(dir='.', suffix='.jpeg') as f:
            f.write(uploaded_file.getbuffer())
            response = query(f.name)
            # st.text(f"The sentiment analysis for ` {user_input} ` is: {result}")
        try:
            st.text(response[0]["generated_text"])
        except Exception as e:
            st.text(str(e))
        
       
with module2_tab:
    import requests

    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    API_TOKEN='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'
    headers = {"Authorization": f"Bearer {API_TOKEN}"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.content
    image_bytes = query({
        "inputs": st.text_input("Enter some text:", "A majestic lion jumping from a big stone at night"),
    })
    # You can access the image with PIL.Image for example
    import io
    from PIL import Image
    image = Image.open(io.BytesIO(image_bytes))
    st.image(image, caption="Selected Image", use_column_width=True)