import json
import os
import openai
import pandas as pd
import numpy as np
import streamlit as st
from openai import OpenAI
from aiconfig import AIConfigRuntime, InferenceOptions, CallbackManager
import asyncio


# Load the aiconfig
config = AIConfigRuntime.load('trend_analysis_model.aiconfig.json')
config.callback_manager = CallbackManager([])



# # Obtain API Key
# def validate_api_key(api_key):
#     return api_keys

# if "api_key" not in st.session_state:
#     api_key = st.text_input("Enter your OpenAI API key", type="password")
#     is_valid = validate_api_key(api_key)
#     if is_valid:
#         st.session_state["api_key"] = api_key
#         st.rerun()
#     elif not is_valid and api_key:
#         st.error("Incorrect or invalid key")
#         st.stop()
#     else:
#         st.info("A Valid OpenAI API key is required")
#         st.stop()
# else:

if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

os.getenv["OPENAI_API_KEY"] = st.session_state["api_key"]

async def get_text_block(uploaded_file, num_rows):
    df = pd.read_csv(uploaded_file).sample(n=num_rows)
    
    inference_options = InferenceOptions(stream=True)
    params = {
        "json_object": json.dumps(df.head().to_dict(orient='records')[0])
    }
    columns_to_drop = await config.run("get_dropped_columns", params, options=inference_options)
    columns_to_drop = columns_to_drop[0].data
    #st.write(columns_to_drop)
    columns_to_drop_clean = [column.strip("[]'") for column in columns_to_drop.split(", ")]
    #st.write(columns_to_drop_clean)
    df = df.drop(columns=columns_to_drop_clean)
    #st.write(df)
    
    # Generate text block
    text_block = ""
    for index, row in df.iterrows():
        text_block += f"{row[df.columns[0]]}\n///\n"
    
    return text_block

async def preprocess_data(text_block):
    inference_options = InferenceOptions(stream=True)
    await config.run("preprocess_steps", options=inference_options)
    params = {
        "dataset_rows": text_block
    }
    preprocess_result = await config.run("get_preprocessed", params, options=inference_options)
    return preprocess_result[0].data
    
async def perform_sentiment_analysis(uploaded_file, num_rows):
    
    inference_options = InferenceOptions(stream=True)
    
    st.session_state.text_block = await get_text_block(uploaded_file,num_rows)
    st.session_state.preprocessed_result = await preprocess_data(st.session_state.text_block)
    
    params = {
        "dataset_rows": st.session_state.text_block
    }
    await config.run("preprocess_steps", options=inference_options)
    result = await config.run("get_analysis", params, options=inference_options)
    metrics = await config.run("get_metrics", options=inference_options)
    return result[0].data, json.loads(metrics[0].data)



# Set the app title
st.title("🐙 Trend Analysis with AIConfig")

# App description
st.write("""
This application is designed to utilize AIConfig's Prompt Chaining capabilities to perform sentiment analysis on any structured dataset.
By using various prompt engineering strategies, the model is able to clean and preprocess csv files, filtering out any columns that 
would not be relevant for sentiment analysis. The model then picks random entries in the dataset - up to a number specified - and performs
accurate sentiment analysis, returning to the user a summary of its findings along with some basic metrics.
""")
st.header('')

# Initialize Session States
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'file_replaced' not in st.session_state:
    st.session_state.file_replaced = False
if 'sidebar_content' not in st.session_state:
    st.session_state.sidebar_content = False
if 'text_block' not in st.session_state:
    st.session_state.text_block = None
if 'preprocessed_result' not in st.session_state:
    st.session_state.preprocessed_result = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'analysis_metrics' not in st.session_state:
    st.session_state.analysis_metrics = None
    
    
# st.write(st.session_state.analysis_metrics)
# st.write(st.session_state.analysis_result)
    
# File Upload Button
left_col, right_col = st.columns(2)
with left_col:
    uploaded_file = st.file_uploader('Upload CSV', type=['csv'], accept_multiple_files=False)
with right_col:
    num_rows = st.number_input('Enter number of samples:', step=1)

if num_rows < 0:
    st.error("Enter a number greater than 0")
elif num_rows > 0:
    if uploaded_file is not None:
        # Check if the uploaded file is new or not
        if st.session_state.uploaded_file is None:
            st.session_state.uploaded_file = uploaded_file
            st.session_state.file_replaced = False
        elif uploaded_file.name != st.session_state.uploaded_file.name:
            st.session_state.uploaded_file = uploaded_file
            st.session_state.file_replaced = True
        else:
            st.session_state.file_replaced = False
            
        if st.button("Submit", type='primary'):
            # try:
            with st.spinner("Processing the request..."):
                st.session_state.analysis_result, st.session_state.analysis_metrics = asyncio.run(perform_sentiment_analysis(uploaded_file, num_rows))                   
            # except Exception:
            #     st.warning("Cannot process request, please try again")
                
        if (st.session_state.analysis_result is not None):
            st.header('')
            st.write(st.session_state.analysis_result)
            st.caption('Disclaimer: This app is for demonstration purposes only and should not be used as professional advice.')
    else:
        st.button("Submit",disabled=True)



with st.sidebar:
    st.header("Analysis Result") 
    #st.write(st.session_state.uploaded_file.name)
    if st.session_state.uploaded_file is None:
        st.write('Please upload a dataset first!')
    elif st.session_state.analysis_metrics is not None:
        
        col1, col2, col3 = st.sidebar.columns(3)
        col1.write("Positive")
        col2.write("Neutral")
        col3.write("Negative")

        col1, col2, col3 = st.sidebar.columns(3)
        col1.write(f"{st.session_state.analysis_metrics['Positive']} rows")
        col2.write(f"{st.session_state.analysis_metrics['Neutral']} rows")
        col3.write(f"{st.session_state.analysis_metrics['Negative']} rows")

        col1, col2, col3 = st.sidebar.columns(3)
        col1.write(f"{round((st.session_state.analysis_metrics['Positive'] / st.session_state.analysis_metrics['Count']) * 100, 1)}%")
        col2.write(f"{round((st.session_state.analysis_metrics['Neutral'] / st.session_state.analysis_metrics['Count']) * 100, 1)}%")
        col3.write(f"{round((st.session_state.analysis_metrics['Negative'] / st.session_state.analysis_metrics['Count']) * 100, 1)}%")


    
    else:
        try:
            with st.spinner('Processing the request...'):
                st.write("Running Sentiment Analysis!")
                # output = compare_resume_to_job_description(job_description_text, resume_text)
            st.write("Done!")           
        except:
            st.warning("Cannot process request now, please try again later")