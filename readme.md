# README

FXP Backend is the backend API endpoint server for the Tiktok 2024 Hackathon. It is a GenAi powered application that enables smart video to text to audio functionality for short form videos on tiktok.

## Description
The `fxp-backend` project is built using FastAPI, a modern, fast (high-performance), web framework for building APIs with Python 3.7+.
Middleware includes, gpt api endpoints, gemini endpoint, amazon boto3 SDK, mongoDb

# .env
Below is what your .env should contain...
gptApiKey = 'gptKey'
geminiKey = 'geminiKey'
s3_api_key = 's3APIAccessKey'
s3_api_secret = 's3SecretKey'
mongoDbUrl = 'mongoDbUrl'

## Installation
To install and run the project, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/fxp-backend.git`
2. Navigate to the project directory: `cd fxp-backend`
3. Create a virtual environment: `python -m venv venv`
4. Activate the virtual environment:
    - On Windows: `venv\Scripts\activate`
    - On macOS and Linux: `source venv/bin/activate`
5. Install the project dependencies: `pip install -r requirements.txt`
6. Start the server: `uvicorn main:app --reload` or `py main.py`

## Usage
Once the server is running, you can access the API endpoints using the following base URL: `http://localhost:8000`
# Initialisation
On first init, call create-user endpoint immediately to create your new user for future usage with all endpoints.
Remember to whitelist your Ip on your mongodb project.
Remember to have database admin access for your s3 bucket user

## API Documentation
The API documentation is automatically generated using FastAPI's built-in capabilities. You can access the documentation by visiting the following URL: `http://localhost:8000/docs`

## Depedencies
fastapi,uvicorn
bcrypt
pymongo
boto3
openai
opencv-python-headless
scikit-image
google-generativeai
dotenv
