import pandas as pd
import numpy as np
import librosa
import streamlit as st

from functions import model_history

st.title("Documentation")
st.divider()
st.subheader("Dataset")
# st.divider()
st.subheader("RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)")
st.write("https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio")
st.write("This portion of the RAVDESS contains 1440 files: 60 trials per actor x 24 actors = 1440. The RAVDESS contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. Speech emotions includes calm, happy, sad, angry, fearful, surprise, and disgust expressions. Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression.")
st.write("Angry:")
st.audio('audio-files/ravdess_angry.wav', format='audio/wav')
st.write("Sad:")
st.audio('audio-files/ravdess_sad.wav', format='audio/wav')

st.subheader("TESS (Toronto Emotional Speech Set)")
st.write("https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess/data")
st.write("There are a set of 200 target words were spoken in the carrier phrase 'Say the word _' by two actresses (aged 26 and 64 years) and recordings were made of the set portraying each of seven emotions (anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral). There are 2800 data points (audio files) in total.")
st.write("The dataset is organised such that each of the two female actor and their emotions are contain within its own folder. And within that, all 200 target words audio file can be found. The format of the audio file is a WAV format")
st.write("Angry:")
st.audio('audio-files/tess_angry.wav', format='audio/wav')
st.write("Sad:")
st.audio('audio-files/tess_sad.wav', format='audio/wav')


st.divider()

st.subheader("Extracted features from audio dataset")
st.write("The result of extracting features from the audio such as MFCC, Chroma STFT, Mel-spectogram, Zero Crossing Rate, and RMSE.")
df = pd.read_csv('csv/ravdess_+_tess.csv')
st.write(df.head())

st.divider()

st.subheader("Model evaluation")
model_history()

st.divider()