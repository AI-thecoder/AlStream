import streamlit as st

st.title("HuggingFace App 🤗")
# # Use a pipeline as a high-level helper
# from transformers import pipeline

# pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

# st.title("HuggingFace App 🤗")
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
# # Check if an image was uploaded
# if uploaded_file is not None:
#     # Display the selected image
#     st.image(uploaded_file, caption="Selected Image", use_column_width=True)

# #user_input = st.text_input("Enter some text:", "")

# #response = classifier(user_input)
# #result=response[0].get('label')


# from PIL import Image

# if uploaded_file is not None:
#   image = Image.open(uploaded_file)
#   response = pipe(image)[0].get('generated_text')
#   #st.text(f"The sentiment analysis for ` {user_input} ` is: {result}")
#   st.text({response})