import streamlit as st
import numpy as np
import pandas as pd
from transformers import pipeline


classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0")

def predict_teacher_course(feedback):
    sequence_to_classify = feedback
    candidate_labels = ["teacher", "course"]
    output = classifier(sequence_to_classify, candidate_labels, multi_label=False)
    return str(output['labels'][0])

def predict_sentiment(feedback):
    sequence_to_classify = feedback
    candidate_labels = ["positive", "negative", "neutral"]
    output = classifier(sequence_to_classify, candidate_labels, multi_label=False)
    return str(output['labels'][0])

def predict_teacher_aspect(feedback):
    sequence_to_classify = feedback
    candidate_labels = ['general', 'teaching skills', 'behaviour', 'knowledge', 'experience', 'assessment']
    output = classifier(sequence_to_classify, candidate_labels, multi_label=False)
    return str(output['labels'][0])

def predict_course_aspect(feedback):
    sequence_to_classify = feedback
    candidate_labels = ['relevancy', 'general', 'content', 'learning material', 'pace']
    output = classifier(sequence_to_classify, candidate_labels, multi_label=False)
    return str(output['labels'][0])

# Streamlit app layout
st.set_page_config(page_title="Aspect-based Sentiment Anlaysis of Student Feedback", layout="centered", initial_sidebar_state="auto")







