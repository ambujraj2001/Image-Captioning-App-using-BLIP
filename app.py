import streamlit as st
import requests
import pickle
from PIL import Image
from transformers import BlipProcessor
with st.sidebar:
    st.subheader('Image Captioning App using BLIP')
    st.write('This app uses the BLIP model to generate captions for images.Model card for image captioning pretrained on COCO dataset - base architecture (with ViT base backbone).')
    image = Image.open('details.png')
    st.image(image, caption='BLIP Model')
    st.code('App Built by Ambuj Raj', language='python')


st.title('Image Captioning App using BLIP')

flag_image=0
flag_url=0
tab1, tab2 = st.tabs(["Upload Image", "Use URL"])
with tab1:
    flag_image=1
    uploaded_file = st.file_uploader("Choose a image",type=['png','jpeg','jpg'])
    if uploaded_file is not None:
        st.image(uploaded_file, width=300)
        raw_image = Image.open(uploaded_file).convert('RGB')

with tab2:
    flag_url=1
    img_url = st.text_input('Enter URL of image')
    if img_url:
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
        st.image(raw_image, width=300)

if st.button('Generate Caption'):
    if(flag_image==1):
        flag_image=1
        with st.spinner('Generating Caption...'):
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            filename = 'blip.sav'
            loaded_model = pickle.load(open(filename, 'rb'))
        inputs = processor(raw_image, return_tensors="pt")
        out = loaded_model.generate(**inputs)
        st.success('Caption Generated!')
        st.write('Generated Caption is: ',processor.decode(out[0], skip_special_tokens=True))
    elif(flag_url==1):
        flag_url=0
        with st.spinner('Generating Caption...'):
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            filename = 'blip.sav'
            loaded_model = pickle.load(open(filename, 'rb'))
        inputs = processor(raw_image, return_tensors="pt")
        out = loaded_model.generate(**inputs)
        st.success('Caption Generated!')
        st.write('Generated Caption is: ',processor.decode(out[0], skip_special_tokens=True))



