import streamlit as st
from streamlit.proto.NumberInput_pb2 import NumberInput
from model import predict
import joblib
import os
import numpy as np

st.set_page_config(page_title="Reddit Post Popularity",
                   layout="wide")

curr_path = os.path.dirname(os.path.realpath(__file__))


feature_cols = joblib.load(curr_path + "/features.joblib")

with st.form("prediction_form"):
    st.header("Enter the Details")
    link = st.text_input("URL of Reddit post:")
    upvote_ratio = st.number_input("Upvote Ratio: ")
    Gilded = st.number_input("Number of awards ", value=0, format="%d")
    num_comments = st.number_input("Number of comments: ")
    upvotes = st.number_input(
        "Number of upvotes: ", value=0, format="%d")
    submit_val = st.form_submit_button("Predict")


if submit_val:

    base_feats = np.array([upvote_ratio, Gilded, num_comments, upvotes])

    attributes = np.concatenate([base_feats])

    print("Attributes value")

    status = predict(base_feats.reshape(1, -1))

    if status:
        st.error("The post is popular")
    else:
        st.success("The App is not popular")
        st.balloons()
