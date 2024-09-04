import streamlit as st
import pandas as pd
import numpy as np
from st_audiorec import st_audiorec
import tempfile

from functions import generate_df, predict_result, save_uploaded_file, process_audio, user_evaluation

acting_script = pd.read_csv('csv/acting-scripts.csv')

# Initialize session state variables
shuffle = st.button("Shuffle")
st.write("Act on following script and emotions!")
if ('rand_script' not in st.session_state) or shuffle:
    acting_script = pd.read_csv('csv/acting-scripts.csv')
    rand_row = acting_script.sample()
    st.session_state.rand_script = rand_row["script"].values[0]
    st.session_state.rand_emotion = rand_row["emotion"].values[0]

# Use session state variables
st.title(st.session_state.rand_script)
st.write("Emotion: ", st.session_state.rand_emotion)
st.divider()
st.subheader("Record audio")
audio_record = st_audiorec()
st.write("or")
st.subheader("Upload audio")
audio_upload = st.file_uploader("Upload file", type=['wav', 'mp3', 'flac'])

st.divider()

# Process uploaded file
if audio_upload:
    st.write("Your audio:")
    st.audio(audio_upload, format='audio/wav')
    predict = st.button("Proceed")
    st.divider()
    if predict:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(audio_upload.read())
            temp_file_path = temp_file.name
        result, accuracy = process_audio(temp_file_path)
        st.write("Result:")
        st.subheader(f"{result} (Accuracy: {accuracy:.2f}%)")
        user_evaluation(result, st.session_state.rand_emotion)

# Process recorded audio
if audio_record:
    st.write("Your audio:")
    st.audio(audio_record)
    predict = st.button("Proceed")
    st.divider()
    if predict:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_record)
            temp_file_path = temp_file.name
        result, accuracy = process_audio(temp_file_path)
        st.write("Result:")
        st.subheader(f"{result} (Accuracy: {accuracy:.2f}%)")
        user_evaluation(result, st.session_state.rand_emotion)
