import streamlit as st
import streamlit_authenticator as stauth

import requests
import base64
import tempfile
import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from openai import OpenAI

# Local EndPoint
API_ENDPOINT = "http://127.0.0.1:8000"

def main():

    if 'isLoggedIn' not in st.session_state:
        st.session_state['isLoggedIn'] = False
    st.title("Audio Sight By Future X Past ")
    username = st.text_input("Enter your username")
    password = st.text_input("Enter your password", type="password")
    if st.button("Login"):
        response = requests.post(f"{API_ENDPOINT}/login", json={"username": username, "password": password})
        if response.status_code == 200:
            st.success("Logged in successfully")
            st.session_state['isLoggedIn'] = True
        else:
            st.error("Invalid username or password")
    if st.session_state['isLoggedIn']:
        # Get User Videos
        videoIds = requests.get(f"{API_ENDPOINT}/list-videos?username={username}").json()
        for videoDict in videoIds['videos']:
            st.title(videoDict['videoName'])
            st.text(videoDict['s3ObjectKey'])
            c1,c2,c3 = st.columns(3)

            editBtn = c1.button('Edit Video Title', key=videoDict['s3ObjectKey'] + 'editBtn')
            deleteBtn = c2.button('Delete Video', key=videoDict['s3ObjectKey'] + 'deleteBtn')
            # Get First Keyframe
            keyframe = requests.get(f"{API_ENDPOINT}/get-keyframe-for-video-base64?username={username}&s3ObjectKey={videoDict['s3ObjectKey']}&keyframeIndex={0}").json()
            if 'imageBase64' in keyframe:
                st.caption('Video Keyframe')
                st.image(keyframe['imageBase64'], width=200)
            if editBtn:
                st.text("Edit Video Title")
                st.text_input("Input New Title")
                saveBtn = st.button("Save", key=videoDict['s3ObjectKey'] + 'saveBtn')
                if saveBtn:
                    response = requests.put(f"{API_ENDPOINT}/update-video-title", json={"username": username, "s3ObjectKey": videoDict['s3ObjectKey'], "newTitle": "New Title"})
                    st.success("Title Saved")

if __name__ == "__main__":
    pg = st.navigation([st.Page("app.py"), st.Page("pages/EditVideoTitle.py")])
    pg.run()
    main()


