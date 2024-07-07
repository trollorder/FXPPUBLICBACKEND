import streamlit as st


def main():
    st.title("Edit Video Title")

    # Get the video path and new title from the user
    video_path = st.text_input("Video Path")
    new_title = st.text_input("New Title")


if __name__ == "__main__":
    main()