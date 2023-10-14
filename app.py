import streamlit as st
# from transformers import pipeline
import requests
from tempfile import NamedTemporaryFile
from PIL import Image
import pandas as pd
import numpy as np
import random
import dicomreader as dcmrdr
import SimpleITK as sitk
import matplotlib.pyplot as plt


# Data Visualization
# import plotly.express as px
import plotly.graph_objs as go
# import plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
# from IPython.display import display
# from plotly.offline import init_notebook_mode
# init_notebook_mode(connected=True)

plotly_template = 'simple_white'

def plot_correlation(org_df):
    '''
    This function is resposible to plot a correlation map among features in the dataset
    '''
    df = org_df.select_dtypes(include='number')
    corr = np.round(df.corr(), 2)
    mask = np.triu(np.ones_like(corr, dtype = bool))
    c_mask = np.where(~mask, corr, 100)

    c = []
    for i in c_mask.tolist()[1:]:
        c.append([x for x in i if x != 100])
    
    fig = ff.create_annotated_heatmap(z=c[::-1],
                                      x=corr.index.tolist()[:-1],
                                      y=corr.columns.tolist()[1:][::-1],
                                      colorscale = 'bluyl')

    fig.update_layout(title = {'text': '<b>Feature Correlation <br> <sup>Heatmap</sup></b>'},
                      height = 650, width = 650,
                      margin = dict(t=210, l = 80),
                      template = 'simple_white',
                      yaxis = dict(autorange = 'reversed'))

    fig.add_trace(go.Heatmap(z = c[::-1],
                             colorscale = 'bluyl',
                             showscale = True,
                             visible = False))
    fig.data[1].visible = True

    return fig

def plot_histogram_matrix(df):
    
    '''
    This function identifies all continuous features within the dataset and plots
    a matrix of histograms for each attribute
    '''
    
    continuous_features = []
    for feat in df.columns:
        if df[feat].nunique() > 2:
            continuous_features.append(feat)
    num_cols = 2
    num_rows = (len(continuous_features) + 1) // num_cols

    fig = make_subplots(rows=num_rows, cols=num_cols)

    for i, feature in enumerate(continuous_features):
        row = i // num_cols + 1
        col = i % num_cols + 1

        fig.add_trace(
            go.Histogram(
                x=df[feature],
                name=feature
            ),
            row=row,
            col=col
        )

        fig.update_xaxes(title_text=feature, row=row, col=col)
        fig.update_yaxes(title_text='Frequency', row=row, col=col)
        fig.update_layout(
            title=f'<b>Histogram Matrix<br> <sup> Continuous Features</sup></b>',
            showlegend=False
        )

    fig.update_layout(
        height=350 * num_rows,
        width=1000,
        margin=dict(t=100, l=80),
        template= plotly_template
    )

    return fig

home_tab, module1_tab, module2_tab,  plot_correlation_tab, api_call_tab, dicom_viewer = st.tabs(["home", "Image Captioning", "Description to Image", "Plot Correlation", "API Call", "Dicom Viewer"])
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

with plot_correlation_tab:
    uploaded_file = st.file_uploader("Import a CSV file", type=["csv"])
    if uploaded_file is not None:
        with NamedTemporaryFile(dir='.', suffix='.jpeg') as f:
            f.write(uploaded_file.getbuffer())
        try:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df)
            # st.data_editor(df)
            st.plotly_chart(plot_correlation(df), theme="streamlit")
            st.plotly_chart(plot_histogram_matrix(df), theme="streamlit")
        except Exception as e:
            st.text(str(e))
    
    
with api_call_tab:
    for i in random.sample(range(1, 101), 3):
        # api_url = "https://jsonplaceholder.typicode.com/todos/"+str(i)
        
        api_url = "https://picsum.photos/id/" + str(i) + "/info"
        response = requests.get(api_url)
        try:
            data_ = response.json()
            st.image(data_["download_url"])
        except Exception as e:
            st.text(str(e))
    # st.json(data)
    # st.write(data['title'])
    # st.write(data['completed'])
    # st.write(response.json())
    # st.write(response.status_code)
    # st.write(response.headers)
    # st.write(response.text)
    # api_url = "https://jsonplaceholder.typicode.com/todos/1"  # Replace with your API endpoint
    # response = requests.get(api_url)

    # if response.status_code == 200:
    #     data = response.json()
    #     df = pd.DataFrame(data)
        
    #     output_file = "api_data.csv"  # Choose a filename for your CSV file
    #     df.to_csv(output_file, index=False)  # Export the DataFrame to a CSV file

    #     print(f"Data exported to {output_file}")
    # else:
    #     print("API request failed with status code:", response.status_code)

with dicom_viewer:
    st.title('DieSitCom')
    dirname = dcmrdr.dir_selector()
    if dirname is not None:
        try:
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(dirname)
            reader.SetFileNames(dicom_names)
            reader.LoadPrivateTagsOn()
            reader.MetaDataDictionaryArrayUpdateOn()
            data = reader.Execute()
            img = sitk.GetArrayViewFromImage(data)
        
            n_slices = img.shape[0]
            slice_ix = st.slider('Slice', 0, n_slices, int(n_slices/2))
            output = st.radio('Output', ['Image', 'Metadata'], index=0)
            if output == 'Image':
                fig = dcmrdr.plot_slice(img, slice_ix)
                st.pyplot(fig)
            else:
                metadata = dict()
                for k in reader.GetMetaDataKeys(slice_ix):
                    metadata[k] = reader.GetMetaData(slice_ix, k)
                df = pd.DataFrame.from_dict(metadata, orient='index', columns=['Value'])
                st.dataframe(df)
        except RuntimeError:
            st.text('This does not look like a DICOM folder!')