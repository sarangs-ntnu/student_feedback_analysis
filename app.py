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

st.markdown("""
#### This application analyzes the student feedback to determine whether it is about a teacher or a course, detects sentiment, and identifies important teacher or course aspects.
""")

# Get user input
user_input = st.text_area("Enter the feedback or comments for analysis:", height=200)

if st.button("Analyze Text"):
    if user_input.strip():
        # Predict whether it's about teacher or course
        type_result = predict_teacher_course(user_input)
        sentiment_result = predict_sentiment(user_input)
        if type_result == 'teacher':
            aspect_result = predict_teacher_aspect(user_input)
        else:
            aspect_result = predict_course_aspect(user_input)

        # Display the results in a nice way
        st.subheader("Analysis Results")

        st.markdown(f"**Type:** `{type_result}`")
        
        st.markdown(f"**Sentiment:** `{sentiment_result}`")
        
        st.markdown("**Aspect Identified:**")
        st.write(", ".join(aspect_result))
    else:
        st.error("Please enter some text for analysis.")

# Add a footer
st.markdown("---")
st.markdown("**Developed by Sarang Shaikh**")
st.markdown("""
Feel free to reach out for more information or suggestions!
""")







