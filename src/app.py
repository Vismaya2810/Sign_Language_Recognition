import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np
import tempfile
import joblib
from tensorflow.keras.models import load_model
from src.data_prep import extract_hand_landmarks
import pandas as pd
from src.nlp_utils import words_to_sentence, correct_grammar
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Sign Language to Text')

# Sidebar model selection
model_option = st.sidebar.radio('Select model(s) to use:', ['CNN', 'LSTM', 'Both'])

# Load models and label encoder
@st.cache_resource
def load_cnn_model():
    model = load_model('src/sign_word_cnn.h5')
    le = joblib.load('src/label_encoder.joblib')
    return model, le

@st.cache_resource
def load_lstm_model():
    model = load_model('src/sign_word_lstm.h5')
    le = joblib.load('src/label_encoder.joblib')
    return model, le

cnn_model, le = load_cnn_model()
lstm_model, le_lstm = load_lstm_model()
SEQ_LEN = 15

# Initialize session state for uploaded files and predictions
if 'uploaded_files' not in st.session_state:
    st.session_state['uploaded_files'] = []
if 'predicted_words_cnn' not in st.session_state:
    st.session_state['predicted_words_cnn'] = []
if 'predicted_words_lstm' not in st.session_state:
    st.session_state['predicted_words_lstm'] = []
if 'show_results' not in st.session_state:
    st.session_state['show_results'] = False

uploaded_files = st.file_uploader('Upload one or more sign language videos (mp4)', type=['mp4'], accept_multiple_files=True)

# If new files are uploaded, reset predictions and flag
if uploaded_files:
    st.session_state['uploaded_files'] = uploaded_files
    st.session_state['predicted_words_cnn'] = []
    st.session_state['predicted_words_lstm'] = []
    st.session_state['show_results'] = False

# Button to mark the end of sign uploads
if st.button('Done - Predict'):
    predicted_words_cnn = []
    predicted_words_lstm = []
    all_probs_cnn = []
    all_probs_lstm = []
    with st.spinner("Running predictions..."):
        for uploaded_file in st.session_state['uploaded_files']:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            tfile.close()
            st.info(f'Extracting hand landmark sequence for {uploaded_file.name}...')
            landmarks_list = extract_hand_landmarks(tfile.name, max_frames=SEQ_LEN)
            if not landmarks_list:
                st.error(f'No hand landmarks detected in {uploaded_file.name}!')
                predicted_words_cnn.append('')
                predicted_words_lstm.append('')
                all_probs_cnn.append(None)
                all_probs_lstm.append(None)
            else:
                st.success(f'Landmarks extracted for {uploaded_file.name}!')
                seq = [l for _, l in landmarks_list]
                if len(seq) < SEQ_LEN:
                    pad = [np.zeros_like(seq[0])] * (SEQ_LEN - len(seq))
                    seq.extend(pad)
                else:
                    seq = seq[:SEQ_LEN]
                seq = np.stack(seq)[None, ...]  # shape (1, SEQ_LEN, 63)
                # CNN prediction
                if model_option in ['CNN', 'Both']:
                    probs_cnn = cnn_model.predict(seq)[0]
                    idx_cnn = np.argmax(probs_cnn)
                    word_cnn = le.inverse_transform([idx_cnn])[0]
                    predicted_words_cnn.append(word_cnn)
                    all_probs_cnn.append(probs_cnn)
                else:
                    predicted_words_cnn.append('')
                    all_probs_cnn.append(None)
                # LSTM prediction
                if model_option in ['LSTM', 'Both']:
                    if lstm_model is not None:
                        probs_lstm = lstm_model.predict(seq)[0]
                        idx_lstm = np.argmax(probs_lstm)
                        word_lstm = le_lstm.inverse_transform([idx_lstm])[0]
                        predicted_words_lstm.append(word_lstm)
                        all_probs_lstm.append(probs_lstm)
                    else:
                        predicted_words_lstm.append('')
                        all_probs_lstm.append(None)
                else:
                    predicted_words_lstm.append('')
                    all_probs_lstm.append(None)
    st.session_state['predicted_words_cnn'] = predicted_words_cnn
    st.session_state['predicted_words_lstm'] = predicted_words_lstm
    st.session_state['all_probs_cnn'] = all_probs_cnn
    st.session_state['all_probs_lstm'] = all_probs_lstm
    st.session_state['show_results'] = True

# Display results only after prediction and only if there are valid words
if st.session_state.get('show_results', False):
    valid_words_cnn = [w for w in st.session_state['predicted_words_cnn'] if w]
    valid_words_lstm = [w for w in st.session_state['predicted_words_lstm'] if w]
    if model_option == 'CNN':
        if valid_words_cnn:
            sentence = words_to_sentence(valid_words_cnn)
            sentence = correct_grammar(sentence)
            st.header('Predicted Words (CNN)')
            st.write(valid_words_cnn)
            st.header('Predicted Sentence (CNN)')
            st.write(sentence)
            st.markdown(f"<h2 style='color: green;'>Predicted Sentence: <b>{sentence}</b></h2>", unsafe_allow_html=True)
            # Show probability bar charts for each video
            for i, probs in enumerate(st.session_state['all_probs_cnn']):
                if probs is not None:
                    labels = le.inverse_transform(np.arange(len(probs)))
                    df = pd.DataFrame({'Probability': probs}, index=labels)
                    st.subheader(f'Probabilities for {st.session_state["uploaded_files"][i].name} (CNN)')
                    st.bar_chart(df)
        else:
            st.warning("No valid predictions were made. Please check your videos.")
    elif model_option == 'LSTM':
        if valid_words_lstm:
            sentence = words_to_sentence(valid_words_lstm)
            sentence = correct_grammar(sentence)
            st.header('Predicted Words (LSTM)')
            st.write(valid_words_lstm)
            st.header('Predicted Sentence (LSTM)')
            st.write(sentence)
            st.markdown(f"<h2 style='color: blue;'>Predicted Sentence: <b>{sentence}</b></h2>", unsafe_allow_html=True)
            for i, probs in enumerate(st.session_state['all_probs_lstm']):
                if probs is not None:
                    labels = le_lstm.inverse_transform(np.arange(len(probs)))
                    df = pd.DataFrame({'Probability': probs}, index=labels)
                    st.subheader(f'Probabilities for {st.session_state["uploaded_files"][i].name} (LSTM)')
                    st.bar_chart(df)
        else:
            st.warning("No valid predictions were made. Please check your videos.")
    elif model_option == 'Both':
        if valid_words_cnn or valid_words_lstm:
            st.header('Predicted Words')
            df_words = pd.DataFrame({
                'Video': [f.name for f in st.session_state['uploaded_files']],
                'CNN': st.session_state['predicted_words_cnn'],
                'LSTM': st.session_state['predicted_words_lstm']
            })
            st.dataframe(df_words)
            # Sentences
            sentence_cnn = words_to_sentence(valid_words_cnn)
            sentence_cnn = correct_grammar(sentence_cnn)
            sentence_lstm = words_to_sentence(valid_words_lstm)
            sentence_lstm = correct_grammar(sentence_lstm)
            st.header('Predicted Sentences')
            if sentence_cnn.strip():
                st.markdown(f"<h3 style='color: green;'>CNN: <b>{sentence_cnn}</b></h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='color: green;'>CNN: <i>No sentence could be formed.</i></h3>", unsafe_allow_html=True)
            if sentence_lstm.strip():
                st.markdown(f"<h3 style='color: blue;'>LSTM: <b>{sentence_lstm}</b></h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='color: blue;'>LSTM: <i>No sentence could be formed.</i></h3>", unsafe_allow_html=True)
            # Side-by-side bar charts for each video
            for i, (probs_cnn, probs_lstm) in enumerate(zip(st.session_state['all_probs_cnn'], st.session_state['all_probs_lstm'])):
                if probs_cnn is not None and probs_lstm is not None:
                    labels = le.inverse_transform(np.arange(len(probs_cnn)))
                    df = pd.DataFrame({
                        'CNN': probs_cnn,
                        'LSTM': probs_lstm
                    }, index=labels)
                    st.subheader(f'Probabilities for {st.session_state["uploaded_files"][i].name}')
                    st.bar_chart(df)
        else:
            st.warning("No valid predictions were made. Please check your videos.") 
