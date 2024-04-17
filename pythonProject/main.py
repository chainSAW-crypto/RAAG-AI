import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import classes
import os
import base64


def main():
    # Background image path
    background_image_path = "C:/Users/TANAY/Downloads/interfacee.jpg"

    # Read the background image and encode it as base64
    with open(background_image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    # Set the background image using CSS
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/webp;base64,{encoded_string}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

    # Custom CSS styles
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Holtwood+One+SC&display=swap');

        body {
            font-family: 'Holtwood One SC', serif;
            background-color: #FFDBAC;
        }

        .title {
            font-size: 48px;
            font-weight: bold;
            color: #e5de00;
            text-align: center;
            margin-top: 50px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
            font-family: 'Holtwood One SC', serif;
        }

        .file-input label {
            font-size: 20px !important; /* Adjust the font size as needed */
        }

        .sidebar-title {
            font-size: 30px;
            font-weight: bold;
            color: #8B4513;
            margin-bottom: 10px;
        }

        .sidebar-file {
            font-size: 24px;
            margin-bottom: 5px;
        }

        .stButton button {
            background-color: #8B4513;
            color: #FFFFFF;
        }

        .stTextInput input {
            background-color: #F5DEB3;
            color: #8B4513;
        }

        .stMarkdown {
            font-family: 'Holtwood One SC', serif;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Use Markdown syntax to style the text with custom CSS
    st.markdown(
        "<p class='title'>Raga Classification AI</p>",
        unsafe_allow_html=True
    )

    # File upload
    st.markdown(
        "<p style='font-size: 24px; font-weight: bold; text-align: left;'>Choose an audio file</p>",
        unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader("__ Audio File**", type=['wav', 'mp3'])

    # Predict button
    predict_button = st.button("Predict")

    if uploaded_file is not None and predict_button:
        predicted_raag = classes.predict_raga(uploaded_file)
        st.success(f"**<span style='font-size:30px;'>Predicted Raag:</span>** {predicted_raag}")

        # Display audio player
        st.audio(uploaded_file)

    # Sample music files sidebar
    st.sidebar.markdown('<h2 class="sidebar-title">Sample Music Files</h2>', unsafe_allow_html=True)
    sample_folder = "C:/Users/TANAY/Downloads/testRaga"
    sample_files = []
    for root, dirs, files in os.walk(sample_folder):
        for file in files:
            if file.endswith(".wav"):
                sample_files.append(os.path.join(root, file))

    selected_sample = st.sidebar.radio('Select a sample file', sample_files)

    if selected_sample and predict_button:
        predicted_raag = classes.predict_raga(selected_sample)
        st.success(f'Predicted Raag: {predicted_raag}')

        # Display audio player for the selected sample
        st.audio(selected_sample)


if __name__ == '__main__':
    st.set_page_config(page_title='Raag Recognition', layout='wide')
    main()