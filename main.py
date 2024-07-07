
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Body
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import dotenv
import uvicorn
from pymongo import MongoClient
from bson.objectid import ObjectId
import bcrypt
import boto3
import uuid
import os 
from openai import OpenAI
import tempfile
import base64
import cv2
from skimage.metrics import structural_similarity as ssim
import zipfile
from io import BytesIO
import google.generativeai as genai
import os
from pathlib import Path


# Loading Api Keys
dotenv.load_dotenv()

# Gemini Endpoitn configuration
genai.configure(api_key=os.getenv('geminiKey'))
geminiModel = genai.GenerativeModel('gemini-1.5-pro-latest')

# FastAPI instance
app = FastAPI()
# Cors Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

# S3 Bucket connection
api_key = 'AKIATCKAQSTXI4N4HDN6'
api_secret = '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'
s3Client = boto3.client('s3', aws_access_key_id=os.getenv('s3_api_key'), aws_secret_access_key=os.getenv('s3_api_secret'))

# MongoDB connection
client = MongoClient(os.getenv('mongoDbUrl'))
collection = client["fxpbackend"]

# Gpt Endpoint Configuration
## Set the API key
gptClient = OpenAI(api_key=os.getenv('gptApiKey'))

# Testing Routes
@app.get("/")
def read_root():
    return 'Connection Sucess'

#region Video Routes
@app.post("/upload-video")
async def upload_video(username: str, video: UploadFile = File(...), videoName: str = ''):

    # User Validation Checks
    if username is None:
        raise HTTPException(status_code=404, detail="User not found")
    else:
        user = collection['users'].find_one({"username": username})
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        
    # Video Validation Checks
    if video.content_type != "video/mp4":
        return {"error": "File must be an MP4 video"}, 400
    if video.size > 50 * 1024 * 1024:
        return {"error": "File size exceeds the limit of 50MB"}

    try:
        # S3 Upload Video
        videoFileNameNew = f"{user['username']}-{str(uuid.uuid4())}"
        s3ClientVideoPath = f'videos/{videoFileNameNew}'
        s3Client.upload_fileobj(video.file, 'videos-tiktok-backend', s3ClientVideoPath)
        video_url = f"https://videos-tiktok-backend.s3.ap-southeast-1.amazonaws.com/videos/{videoFileNameNew}"
        collection["videos"].insert_one({"videoUrl": video_url, 'belongsTo': user['_id'], 's3ObjectKey' : f'videos/{videoFileNameNew}', 'videoName': videoName})
        return {"message": "Video uploaded successfully", "videoUrl": video_url}
    except Exception as e:
        return {"error": str(e)}, 500

@app.delete("/delete-video")
def delete_video(videoS3ObjectKey: str, username: str):
    # Find user document in MongoDB collection
    user = collection["users"].find_one({"username": username})
    if user is None:
        return {"error": "User not found"}
    
    # Find video document in MongoDB collection
    video = collection["videos"].find_one({"s3ObjectKey": videoS3ObjectKey, 'belongsTo': user["_id"]})
    if video is None:
        return {"error": "Video not found for user"}
    
    # if Video is found check if keyframes exist
    keyframes = list(collection["keyframes"].find({"belongsTo": username, 's3ObjectKey': video['s3ObjectKey']}))
    if len(keyframes) > 0:
        for keyframe in keyframes:
            keyframe_key = f"keyframes/{keyframe['s3ObjectKey'].split('/')[-1]}/{str(keyframe['keyframeIndex'])}.jpg"
            s3Client.delete_object(Bucket='videos-tiktok-backend', Key=keyframe_key)
            collection["keyframes"].delete_one({"_id": keyframe['_id']})
    

    # Delete video from S3 bucket
    video_url = video["videoUrl"]
    video_key = video_url.split("/")[-1]
    s3Client.delete_object(Bucket='videos-tiktok-backend', Key=video_key)

    # Delete video document from MongoDB collection
    collection["videos"].delete_one({"s3ObjectKey": videoS3ObjectKey, 'belongsTo': user["_id"]})   

    return {"message": "Video deleted successfully"}

@app.get("/list-videos")
def list_videos(username: str):
    # Find user document in MongoDB collection
    user = collection["users"].find_one({"username": username})
    if user is None:
        return {"error": "User not found"}

    # Find videos belonging to the user
    print(user['_id'])
    videos = list(collection["videos"].find({"belongsTo": user["_id"]}))
    # clean object id in mongo db
    videos = [{"videoUrl": video["videoUrl"], "s3ObjectKey": video["s3ObjectKey"], "videoName": video["videoName"]} for video in videos]

    return {"videos": videos}

@app.get('/video-details')
def video_details(username: str, s3ObjectKey: str):
    # Find user document in MongoDB collection
    user = collection["users"].find_one({"username": username})
    if user is None:
        return {"error": "User not found"}
    # Find video document in MongoDB collection
    video = collection["videos"].find_one({"belongsTo": user["_id"], "s3ObjectKey": s3ObjectKey})
    if video is None:
        return {"error": "Video not found"}
    return {"videoUrl": video["videoUrl"], "s3ObjectKey": video["s3ObjectKey"], "videoName": video["videoName"]}

@app.put("/update-video-title")
def update_video_title(username: str, s3ObjectKey: str, newTitle:str):
    # validate username
    print(username)
    user = collection["users"].find_one({"username": username})
    if user is None:
        return {"error": "User not found"}
    
    # Find video document in MongoDB collection
    video = collection["videos"].find_one({"s3ObjectKey": s3ObjectKey, 'belongsTo': user["_id"]})
    if video is None:
        return {"error": "Video not found for user"}
    # Update video document with new title
    collection["videos"].update_one({"s3ObjectKey": s3ObjectKey, 'belongsTo': user["_id"]}, {"$set": {"videoName": newTitle}})
    return {"message": "Video title updated successfully"}

# endregion

#region User Routes
@app.post("/create-user")
def create_user(username: str = Body(..., media_type="application/json"), password: str  = Body(..., media_type="application/json"), email: str = Body(..., media_type="application/json")):
    # Generate salted password hash

    salted_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

    # Create user document
    user = {
        "username": username,
        "password": salted_password,
        "email": email
    }

    # Insert user document into MongoDB collection
    collection["users"].insert_one(user)

    return {"message": "User created successfully"}

@app.post("/login")
def login(username: str = Body(..., media_type="application/json"), password: str = Body(..., media_type="application/json")):
    # Find user document in MongoDB collection
    user = collection["users"].find_one({"username": username})
    if user is None:
        return {"error": "Invalid username"}

    # Verify password
    if bcrypt.checkpw(password.encode(), user["password"]):
        return {"message": "Login successful"}
    else:
        return {"error": "Invalid password"}
    
@app.post("/reset-password")
def reset_password(username: str, new_password: str):
    # Find user document in MongoDB collection
    user = collection["users"].find_one({"username": username})
    if user is None:
        return {"error": "Invalid username"}

    # Generate salted password hash
    salted_password = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt())

    # Update user document with new password
    collection["users"].update_one({"username": username}, {"$set": {"password": salted_password}})

    return {"message": "Password reset successful"}

# endregion

#region Transcript Routes
# Each video id will hasve a corresponding transcript
@app.get("/download-transcript")
def download_transcript(username: str, s3ObjectKey: str):
    # validate username
    user = collection["users"].find_one({"username": username})
    if user is None:
        return {"error": "User not found"}
    
    # validate s3ObjectKey belongs to user
    video = collection["videos"].find_one({"belongsTo": user["_id"], "s3ObjectKey": s3ObjectKey})
    if video is None:
        return {"error": "Video does not belong to user"}
    

    # Download assumes keyframes has been generated and descriptions have been stored in mongodb
    keyframes = list(collection["keyframes"].find({"belongsTo": username, "s3ObjectKey": s3ObjectKey}))
    if len(keyframes) == 0:
        return {"error": "Keyframes not found"}
    descriptions = []
    for keyframe in keyframes:
        descriptions.append(keyframe['description'])
    transcript = '\n'.join(descriptions)
    return {"transcript": transcript}
    
# Generate Transcript for s3ObjectKey given a username
@app.post("/generate-transcript_v2")
def generateTranscriptv2(s3ObjectKey : str, username: str, forceGeneration: bool = False):
    # validate username
    user = collection["users"].find_one({"username": username})
    if user is None:
        return {"error": "User not found"}
    # validate s3ObjectKey belongs to user
    video = collection["videos"].find_one({"belongsTo": user["_id"], "s3ObjectKey": s3ObjectKey})
    if video is None:
        return {"error": "Video not found"}
    
    # validate s3ObjectKey does not have descriptions already
    keyframes = list(collection["keyframes"].find({"belongsTo": username, 's3ObjectKey': s3ObjectKey}))
    print(f'Existing keyframes: {len(keyframes)}')
    if len(keyframes) > 0 and not forceGeneration:
        # if keyframes exist return the transcript by combining the key frames if not forceGeneration
        descriptions = []
        for keyframe in keyframes:
            descriptions.append(keyframe['description'])
        transcript = '\n'.join(descriptions)
        return {"transcript": transcript}
    if forceGeneration:
        # if Force Generation is True, delete existing keyframes transcripts
        for keyframe in keyframes:
            collection["keyframes"].delete_one({"_id": keyframe['_id']})
        print(f"Deleted existing keyframe descriptions from MongoDB successfully")
        # Check if audio exists
        audio = collection["audios"].find_one({"belongsTo": user['_id'], 's3ObjectKey': s3ObjectKey})
        if audio is None:
            pass
        else:
            # Delete audio for transcript from s3
            response = s3Client.delete_object(Bucket='videos-tiktok-backend', Key=f"audios/{s3ObjectKey.split('/')[-1]}.mp3")
            collection["audios"].delete_one({"belongsTo": user['_id'], 's3ObjectKey': s3ObjectKey})
            print('audio deleted successfully')

    # if keyframes do not exist, generate the transcript
    # get videoObject
    response = s3Client.get_object(Bucket='videos-tiktok-backend', Key=s3ObjectKey)
    videoBinary = response['Body'].read()
    
    # check if video exists
    if videoBinary is None:
        return {"error": "Video not found in s3 Bucket"}
    print('Generating Transcript')
    transcript = getTranscriptv2(videoBinary,s3ObjectKey, 'kelvin')
    return {"transcript": transcript}

@app.get("/download-keyframes-for-video")
def download_keyframes_for_video(username: str, s3ObjectKey: str):
    # validate username
    user = collection["users"].find_one({"username": username})
    if user is None:
        return {"error": "User not found"}
    #  validate s3ObjectKey belongs to user
    video = collection["videos"].find_one({"belongsTo": user["_id"], "s3ObjectKey": s3ObjectKey})
    if video is None:
        return {"error": "Video not found"}
    
    # get keyframes
    keyframes = list(collection["keyframes"].find({"belongsTo": username, 's3ObjectKey': s3ObjectKey}))
    if len(keyframes) == 0:
        return {"error": "Keyframes not found"}
    images=[]    
    # download keyframes from s3
    for idx, keyframe in enumerate(keyframes):
        keyframe_key = f"keyframes/{s3ObjectKey.split('/')[-1]}/{str(keyframes[idx]['keyframeIndex'])}.jpg"
        response = s3Client.get_object(Bucket='videos-tiktok-backend', Key=keyframe_key)
        imageBinary = response['Body'].read()
        images.append(imageBinary)

    # Write image binaries to temporary files
    temp_files = []
    temp_dir = 'temp_keyframes'
    os.makedirs(temp_dir, exist_ok=True)
    for idx, image_binary in enumerate(images):
        with open(f'temp_keyframes/{idx}.jpg' , 'wb') as f:
            f.write(image_binary)
            temp_files.append(f'{temp_dir}/{idx}.jpg')
    print(temp_files)
    # Create a zip file containing the temporary files
    zip_file_name = 'keyframes.zip'
    with zipfile.ZipFile(zip_file_name, 'w') as zip_file:
        for temp_file in temp_files:
            zip_file.write(temp_file, os.path.basename(temp_file))

    # Remove the temporary files
    for temp_file in temp_files:
        os.remove(temp_file)

    response = FileResponse(zip_file_name, media_type='application/zip', filename='Keyframes.zip')
    return response

@app.get("/get-keyframeUrls-for-video")
def get_keyframes_for_video(username: str, s3ObjectKey: str):
    # validate username
    user = collection["users"].find_one({"username": username})
    if user is None:
        return {"error": "User not found"}
    #  validate s3ObjectKey belongs to user
    video = collection["videos"].find_one({"belongsTo": user["_id"], "s3ObjectKey": s3ObjectKey})
    if video is None:
        return {"error": "Video not found"}
    
    # get keyframes
    keyframes = list(collection["keyframes"].find({"belongsTo": username, 's3ObjectKey': s3ObjectKey}))
    if len(keyframes) == 0:
        return {"error": "Keyframes not found"} 
    
    keyframeUrls = [keyframe['imageUrl'] for keyframe in keyframes]
    return {"keyframeUrls": keyframeUrls}
    
@app.post("/evaluate-transcript")
async def evaluate_transcript(transcript: str = Body(..., media_type="text/plain")):
    # Ensure length strictly doesnt exceed 10000 characters
    if len(transcript) > 10000:
        return {"error": "Transcript length exceeds 5000 characters"}
    gptResponse = evaluate_transcript_gpt(transcript)
    geminiResponse = evaluate_transcript_gemini(transcript)
    # aggregate responses
    score = int((int(gptResponse) + int(geminiResponse)) / 2)
    return {"score": score}

@app.post("/transcript-to-mp3")
def transcript_to_mp3(transcript : str = Body(..., media_type="text/plain"), outputName: str = "output.mp3"):
    # Create text-to-speech audio file
    with gptClient.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        input=transcript,
    ) as response:
        response.stream_to_file(outputName)
    with open(outputName, "rb") as f:
        audio_binary = f.read()
    # audio_base64 = base64.b64encode(audio_binary).decode('utf-8')
    # audio_base64_with_headers = f"data:audio/mp3;base64,{audio_base64}"
    
    return FileResponse(outputName, media_type='audio/mp3', filename=outputName)

@app.post("/transcript-to-base64-string")
def transcript_to_base64(transcript : str = Body(..., media_type="text/plain"), outputName: str = "output.mp3"):
    # Create text-to-speech audio file
    with gptClient.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        input=transcript,
    ) as response:
        response.stream_to_file(outputName)
    with open(outputName, "rb") as f:
        audio_binary = f.read()
    audio_base64 = base64.b64encode(audio_binary).decode('utf-8')
    audio_base64_with_headers = f"data:audio/mp3;base64,{audio_base64}"
    
    return {"audioBase64": audio_base64_with_headers}
# endregion

#region Audio Routes
# Generate Audio for Transcript
@app.post("/generate-audio-for-transcript")
def save_mp3(s3ObjectKey: str, username: str, forceGeneration: bool = False):
    # validate username
    user = collection["users"].find_one({"username": username})
    if user is None:
        return {"error": "User not found"}
    #  validate s3ObjectKey belongs to user
    video = collection["videos"].find_one({"belongsTo": user["_id"], "s3ObjectKey": s3ObjectKey})
    if video is None:
        return {"error": "Video not found"}
    
    # get transcript
    keyframes = list(collection["keyframes"].find({"belongsTo": username, 's3ObjectKey': s3ObjectKey}))
    if len(keyframes) == 0:
        return {"error": "Keyframes not found"}
    descriptions = []
    for keyframe in keyframes:
        descriptions.append(keyframe['description'])

    transcript = '\n'.join(descriptions)
    print(transcript)
    
    # check if audio transcript exists
    audio = collection["audios"].find_one({"belongsTo": user['_id'], 's3ObjectKey': s3ObjectKey})
    if audio is not None and not forceGeneration:
        print('Audio already exists')
        response = s3Client.get_object(Bucket='videos-tiktok-backend', Key=f"audios/{s3ObjectKey.split('/')[-1]}.mp3")
        base64Encode = 'data:audio/mp3;base64,' + base64.b64encode(response['Body'].read()).decode('utf-8')  
        return {"audioBase64": base64Encode}
    elif forceGeneration:
        # remove existing audio
        response = s3Client.delete_object(Bucket='videos-tiktok-backend', Key=f"audios/{s3ObjectKey.split('/')[-1]}.mp3")
        collection["audios"].delete_one({"belongsTo": user['_id'], 's3ObjectKey': s3ObjectKey})
        print('Deleted existing audio')
        # continue to generate audio
    
    # # Create text-to-speech audio file
    with gptClient.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        input=transcript[:4000],
    ) as response:
        response.stream_to_file('output.mp3')
    with open('output.mp3', "rb") as f:
        audioBinary = f.read()

    # Save the audio file to s3
    s3Client.put_object(Bucket='videos-tiktok-backend', Key=f"audios/{s3ObjectKey.split('/')[-1]}.mp3", Body=audioBinary)
    collection["audios"].insert_one({"audioUrl": f"https://videos-tiktok-backend.s3.ap-southeast-1.amazonaws.com/audios/{s3ObjectKey.split('/')[-1]}.mp3", 'belongsTo': user['_id'], 's3ObjectKey': s3ObjectKey})
    
    return {"audioBase64": 'data:audio/mp3;base64,' + base64.b64encode(audioBinary).decode('utf-8')}

# Get Audio for Transcript
@app.get("/get-audio-for-transcript")
def get_audio_for_transcript(username: str, s3ObjectKey: str):
    # validate username
    user = collection["users"].find_one({"username": username})
    if user is None:
        return {"error": "User not found"}
    #  validate s3ObjectKey belongs to user
    video = collection["videos"].find_one({"belongsTo": user["_id"], "s3ObjectKey": s3ObjectKey})
    if video is None:
        return {"error": "Video not found"}
    
    # get audio for transcript from s3
    response = s3Client.get_object(Bucket='videos-tiktok-backend', Key=f"audios/{s3ObjectKey.split('/')[-1]}.mp3")
    audioBinary = response['Body'].read()
    audioBase64 = 'data:audio/mp3;base64,' + base64.b64encode(audioBinary).decode('utf-8')
    return {"audioBase64": audioBase64}

# Delete Audio for Transcript
@app.delete("/delete-audio-for-transcript")
def delete_audio_for_transcript(username: str, s3ObjectKey: str):
    # validate username
    user = collection["users"].find_one({"username": username})
    if user is None:
        return {"error": "User not found"}
    #  validate s3ObjectKey belongs to user
    video = collection["videos"].find_one({"belongsTo": user["_id"], "s3ObjectKey": s3ObjectKey})
    if video is None:
        return {"error": "Video not found"}
    # Check if audio exists
    audio = collection["audios"].find_one({"belongsTo": user['_id'], 's3ObjectKey': s3ObjectKey})
    if audio is None:
        return {"error": "Audio not found"}
    
    # Delete audio for transcript from s3
    keyAudio = f"audios/{s3ObjectKey.split('/')[-1]}.mp3"
    print(keyAudio)
    response = s3Client.delete_object(Bucket='videos-tiktok-backend', Key=keyAudio)
    collection["audios"].delete_one({"belongsTo": user['_id'], 's3ObjectKey': s3ObjectKey})
    return {"message": "Audio deleted successfully"}

#endregion

# region Keyframes
# Keyframe and Description Routes
# Get Keyframes for Video in a list of base64 strings
@app.get("/get-all-keyframes-for-video-base64")
def get_all_keyframes_for_video_base64(username: str, s3ObjectKey: str):
    # validate username
    user = collection["users"].find_one({"username": username})
    if user is None:
        return {"error": "User not found"}
    #  validate s3ObjectKey belongs to user
    video = collection["videos"].find_one({"belongsTo": user["_id"], "s3ObjectKey": s3ObjectKey})
    if video is None:
        return {"error": "Video not found"}
    # ensure keyframes exist
    keyframes = list(collection["keyframes"].find({"belongsTo": username, 's3ObjectKey': s3ObjectKey}))
    if len(keyframes) == 0:
        return {"error": "Keyframes not found"}
    images=[]
    # download keyframes from s3
    for idx, keyframe in enumerate(keyframes):
        keyframe_key = f"keyframes/{s3ObjectKey.split('/')[-1]}/{str(keyframes[idx]['keyframeIndex'])}.jpg"
        response = s3Client.get_object(Bucket='videos-tiktok-backend', Key=keyframe_key)
        imageBinary = response['Body'].read()
        images.append(imageBinary)
    # Convert images to base64 strings
    image_base64_strings = ['data:image/jpeg;base64,' + base64.b64encode(image).decode('utf-8') for image in images]

    formattedDict = {x['keyframeIndex']:{'keyframeIndex' : x['keyframeIndex'], 'description' : x['description'], 's3ObjectKey':x['s3ObjectKey'], 'filteredFrameIndexTimestamp' :x['filteredFrameIndexTimestamp']} for x in keyframes}
    for idx, imageBase64 in enumerate(image_base64_strings):
        formattedDict[idx]['imageBase64'] = imageBase64
    return {"keyframes": formattedDict}

@app.get("/get-keyframe-for-video-base64")
def get_keyframeDict_for_video_base64(username: str, s3ObjectKey: str, keyframeIndex: int):
    # validate username
    user = collection["users"].find_one({"username": username})
    if user is None:
        return {"error": "User not found"}
    #  validate s3ObjectKey belongs to user
    video = collection["videos"].find_one({"belongsTo": user["_id"], "s3ObjectKey": s3ObjectKey})
    if video is None:
        return {"error": "Video not found"}
    # ensure keyframes exist
    keyframe = collection["keyframes"].find_one({"belongsTo": username, 's3ObjectKey': s3ObjectKey, 'keyframeIndex': keyframeIndex})
    if keyframe is None:
        return {"error": "Keyframe not found"}
    # download keyframe from s3
    keyframe_key = f"keyframes/{s3ObjectKey.split('/')[-1]}/{str(keyframeIndex)}.jpg"
    response = s3Client.get_object(Bucket='videos-tiktok-backend', Key=keyframe_key)
    imageBinary = response['Body'].read()
    imageBase64 = 'data:image/jpeg;base64,' + base64.b64encode(imageBinary).decode('utf-8')
    return {"keyframeIndex": keyframeIndex, "s3ObjectKey": s3ObjectKey, "description": keyframe['description'],"imageBase64": imageBase64, 'filteredFrameIndexTimestamp' :keyframe['filteredFrameIndexTimestamp']}

# Get Keyframes for Video in a list of URLs
@app.put("/edit-keyframe-description")
def edit_keyframe_description(username: str, s3ObjectKey: str = Body(..., media_type="application/json"), keyframeIndex: int = Body(..., media_type="application/json"), newDescription: str = Body(..., media_type="application/json")):
    # validate username
    user = collection["users"].find_one({"username": username})
    if user is None:
        return {"error": "User not found"}
    #  validate s3ObjectKey belongs to user
    video = collection["videos"].find_one({"belongsTo": user["_id"], "s3ObjectKey": s3ObjectKey})
    if video is None:
        return {"error": "Video not found"}
    # ensure keyframes exist
    keyframe = collection["keyframes"].find_one({"belongsTo": username, 's3ObjectKey': s3ObjectKey, 'keyframeIndex': keyframeIndex})
    if keyframe is None:
        return {"error": "Keyframe not found"}
    
    # Validate newDescription is not empty
    if newDescription == "":
        return {"error": "Description cannot be empty"}
    
    # update keyframe description
    collection["keyframes"].update_one({"_id": keyframe['_id']}, {"$set": {"description": newDescription}})
    return {"message": f"Keyframe updated successfully with description {newDescription}" , "keyframeIndex": keyframeIndex, "s3ObjectKey": s3ObjectKey}

# Given keyframeIndex, regenerate the description of the keyframe using GPT
@app.post('/regenerate-keyframe-description')
def regenerate_keyframe_description(username: str, s3ObjectKey: str= Body(..., media_type="application/json"), keyframeIndex: int= Body(..., media_type="application/json")):
    # validate username
    user = collection["users"].find_one({"username": username})
    if user is None:
        return {"error": "User not found"}
    #  validate s3ObjectKey belongs to user
    video = collection["videos"].find_one({"belongsTo": user["_id"], "s3ObjectKey": s3ObjectKey})
    if video is None:
        return {"error": "Video not found"}
    # ensure keyframes exist
    keyframe = collection["keyframes"].find_one({"belongsTo": username, 's3ObjectKey': s3ObjectKey, 'keyframeIndex': keyframeIndex})
    if keyframe is None:
        return {"error": "Keyframe not found"} 
    # Get original Image from S3
    keyframeBinary = s3Client.get_object(Bucket='videos-tiktok-backend', Key=f"keyframes/{s3ObjectKey.split('/')[-1]}/{str(keyframeIndex)}.jpg")['Body'].read()
    # Regenerate Description
    # 2 Possibilities, if keyframeIndex is 0 then zero shot generation but if it is not 0 then use the previous keyframe description as context for the new description
    if keyframeIndex == 0:
        newDescription = describeImageGPT_v2(keyframeBinary)
    else:
        previousKeyframe = collection["keyframes"].find_one({"belongsTo": username, 's3ObjectKey': s3ObjectKey, 'keyframeIndex': keyframeIndex - 1})
        newDescription = describeImageGPT_v2(keyframeBinary, previousKeyframe['description'])
        
    # Update keyframe description
    collection["keyframes"].update_one({"_id": keyframe['_id']}, {"$set": {"description": newDescription}})
    return {"message": f"Keyframe description regenerated successfully with new description {newDescription}"}

# Delete Keyframe for Video. Once deleted the transcript will not contain the description for that keyframe
@app.delete("/delete-keyframe-for-video")
def delete_keyframe_for_video(username: str, s3ObjectKey: str, keyframeIndex: int):
    # validate username
    user = collection["users"].find_one({"username": username})
    if user is None:
        return {"error": "User not found"}
    #  validate s3ObjectKey belongs to user
    video = collection["videos"].find_one({"belongsTo": user["_id"], "s3ObjectKey": s3ObjectKey})
    if video is None:
        return {"error": "Video not found"}
    # ensure keyframes exist
    keyframe = collection["keyframes"].find_one({"belongsTo": username, 's3ObjectKey': s3ObjectKey, 'keyframeIndex': keyframeIndex})
    if keyframe is None:
        return {"error": "Keyframe not found"}
    # delete keyframe from s3
    keyframe_key = f"keyframes/{s3ObjectKey.split('/')[-1]}/{str(keyframeIndex)}.jpg"
    response = s3Client.delete_object(Bucket='videos-tiktok-backend', Key=keyframe_key)
    # delete keyframe from mongodb
    collection["keyframes"].delete_one({"_id": keyframe['_id']})
    return {"message": "Keyframe deleted successfully"}

# endregion

#region ML/AI Helper Function

# Function to sample unqiue scenes from a video binary
def extract_unique_scenes(video_binary, sampling_seconds=1, ssim_threshold=0.3, autoFineTune=False):
    

    # Write the video binary to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(video_binary)
        temp_video_path = temp_video.name

    #  New Video Obj From Temp File Save
    cap = cv2.VideoCapture(temp_video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps * sampling_seconds)
    
    # Extract frames at the specified interval
    frames = []
    for i in range(0, frame_count, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break

    cap.release()
    os.remove(temp_video_path)
    
    if not frames:
        return []

    # Convert frames to grayscale in place
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]

    # Filter frames using SSIM
    
    

    # Add first frame
    filtered_frames = [frames[0]]  
    filtered_frame_indexes = [0] # Always include the first frame

    # Always compare to the last scanned frame
    lastScannedFrame = 0
    autofinetunePercentage = 0.1
    iterationLimit= 20
    counter = 1
    
    # preference for more detailed description so if last step was incrementing, we will revert the incrementation 
    isLastStepIncrementing = False

    
    if autoFineTune:           
        isFineTuning = True
        optimised_ssim_threshold = ssim_threshold #init ssim threshold
        targetFrames = len(frames) // 3 # so on average 1 new keyframe per 3 seconds
        while isFineTuning:
            # Ensure that last Scanned Frame is 0
            lastScannedFrame = 0
            # Ensure Filtered_frames are reset
            filtered_frames = [frames[0]]
            # Ensure Filtered_frame_indexes are reset
            filtered_frame_indexes = [0]
            # Last Iteration due to Iteration Limit
            if counter > iterationLimit:
                if isLastStepIncrementing:
                    # revert the last incrementation
                    optimised_ssim_threshold -= incrementStep
                # last iteration
                for i in range(1, len(frames)):
                    ssim_score, _ = ssim(gray_frames[lastScannedFrame], gray_frames[i], full=True)
                    if ssim_score < optimised_ssim_threshold:
                        filtered_frames.append(frames[i])
                        filtered_frame_indexes.append(i)
                        lastScannedFrame = i
                print(f'Number of unique scenes: {len(filtered_frames)}, Limit Reached')
                break
            # SSIM comparison
            for i in range(1, len(frames)):
                ssim_score, _ = ssim(gray_frames[lastScannedFrame], gray_frames[i], full=True)
                if ssim_score < optimised_ssim_threshold:
                    filtered_frames.append(frames[i])
                    filtered_frame_indexes.append(i)
                    lastScannedFrame = i
            print(f'Number of unique scenes: {len(filtered_frames)}')

            # Increment and reset Because we are lesser thanthe targetFrames
            if len(filtered_frames) <= targetFrames:
                if max(0,targetFrames-1)>len(filtered_frames):
                    incrementStep = optimised_ssim_threshold * (autofinetunePercentage)
                    print(incrementStep)

                    # reduce autofinetunePrecentage
                    autofinetunePercentage = autofinetunePercentage * 0.9
                    optimised_ssim_threshold += incrementStep
                    print(f'Fine Tuning Incrementing by {round(incrementStep,3)}')
                    counter += 1
                    isLastStepIncrementing = True
                else:
                    isFineTuning = False
            else:
                # Decrease and reset As we are more than the targetFrames
                decrementStep = optimised_ssim_threshold*autofinetunePercentage
                # reduce autofinetunePrecentage
                autofinetunePercentage = autofinetunePercentage * 0.9
                optimised_ssim_threshold -= decrementStep 
                print(f'Fine Tuning Decrementing by {round(decrementStep,3)}')
                counter += 1
                isLastStepIncrementing = False

    
    else:
        # SSIM comparison
        for i in range(1, len(frames)):
            ssim_score, _ = ssim(gray_frames[lastScannedFrame], gray_frames[i], full=True)
            if ssim_score < ssim_threshold:
                filtered_frames.append(frames[i])
                filtered_frame_indexes.append(i)
                lastScannedFrame = i

    # Print the number of unique scenes
    print(f'Number of unique scenes: {len(filtered_frames)}')

    

    # Convert filtered frames to JPG binaries
    jpg_binaries = [cv2.imencode('.jpg', frame)[1].tobytes() for frame in filtered_frames]

    return jpg_binaries , filtered_frame_indexes

# Function to sample frames from a video binary by time deltas ANTIQUATED
def sampleFrames(videoBinary):
    # Write the video binary to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(videoBinary)
        temp_video_path = temp_video.name
    
    # Create a VideoCapture object from the temporary video file
    cap = cv2.VideoCapture(temp_video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame interval for sampling
    sampling_seconds = 5
    # Sampling Seconds
    frame_interval = int(fps * sampling_seconds)
    
    # Sample frames
    sampled_frames = []
    for i in range(0, frame_count, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            # Encode frame to jpg binary
            _, jpg = cv2.imencode('.jpg', frame)
            sampled_frames.append(jpg.tobytes())
    
    cap.release()
    
    # Remove the temporary video file
    import os
    os.remove(temp_video_path)
    
    return sampled_frames

# Function to generate transcript with raw sampling and no previous context ANTIQUATED
def getTranscript(videoBinary, s3ObjectKey, username, additionalContext = None):

    descriptions = []

    # 1) Extract Keyframes

    # Simple Extract Keyframes
    # keyframes = sampleFrames(videoBinary)

    # Smart Extract Keyframes
    keyframes = extract_unique_scenes(videoBinary, autoFineTune=True)
    print(f'Extracted {len(keyframes)} keyframes')

    # Prereq 2 : Store to s3 Bucket 
    for idx, keyframeBinary in enumerate(keyframes):
        keyframepath = f"keyframes/{s3ObjectKey.split('/')[-1]}/{str(idx)}.jpg"
        s3Client.put_object(Bucket='videos-tiktok-backend', Key=keyframepath, Body=keyframeBinary)
        print(f"Uploaded keyframe {keyframes.index(keyframeBinary) + 1} to S3 successfully.")

    # 2) Describe Keyframes
    descriptions = []
    for keyframe in keyframes:
        description =describeImageGPT(keyframe,additionalContext)
        descriptions.append(description)    
        

    # Prereq 3 is to store individual keyframe descriptions to mongodb with associated s3ObjectKey
    for i in range(len(descriptions)):
        imageUrl = f"https://videos-tiktok-backend.s3.ap-southeast-1.amazonaws.com/keyframes/{s3ObjectKey.split('/')[-1]}/{str(i)}.jpg" 
        collection["keyframes"].insert_one({"imageUrl": imageUrl, "keyframeIndex": i, "description": descriptions[i], 'belongsTo' : username, "s3ObjectKey": s3ObjectKey})
        print(f"Inserted keyframe {i + 1} description to MongoDB successfully")
    
    # 3) Return Transcript
    transcript = '\n'.join(descriptions)
    return transcript

# Function Improved to zero shot keyframes
def describeImageGPT(imageBinary, additionalContext = None):
    # Convert image binary to base64 string
    imageBase64 = base64.b64encode(imageBinary).decode('utf-8')

    completion = gptClient.chat.completions.create(
    model='gpt-4o',
    messages=[
            {"role": "system", "content": "You are an expert in describing tiktok frames in an interesting manner for the visually impaired. The images you describe will be used to generate a transcript for a video but never mention anything about tiktok. You can start by describing this image"},
            {"role": "user", "content": [
                {"type": "text", "text": "Describe the image in an interesting and captivating manner that is short and succinct."},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{imageBase64}"}
                }
            ]}
        ],
        temperature=0.7,
    )
    description = completion.choices[0].message.content
    print(description)
    
    return description
# Function Improved to generate with previous frame as context if exist
def describeImageGPT_v2(imageBinary, previousCaption = None ,additionalContext = None):
    # Convert image binary to base64 string
    imageBase64 = base64.b64encode(imageBinary).decode('utf-8')

    if previousCaption:
        formattedMessages = [
                {"role": "system", "content": "You are an expert in describing tiktok frames in an interesting manner for the visually impaired. The images you describe will be used to generate a transcript for a video. Exclude the creator's name and do not mention tiktok. Keep it to less than 30 words per frame. If you see a frame that shows tiktok logo only, you can ignore that frame and just say the video ends"},
                {"role": "user", "content": [
                    {"type": "text", "text": f'There was a previous image and this was the caption: {previousCaption}'},
                    {"type": "text", "text": "Describe the image below which is the next scene after the previous image in an interesting and captivating manner that is short and succinct."},
                    {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{imageBase64}"}
                    },
                    
                ]}
            ]
        if additionalContext:
            formattedMessages[1]['content'].append({"type": "text", "text": additionalContext})
        completion = gptClient.chat.completions.create(
            model='gpt-4o',
            messages=formattedMessages,
                temperature=0.7,
            )
    else:
        formattedMessages = [
                    {"role": "system", "content": "You are an expert in describing tiktok frames in an interesting manner for the visually impaired. The images you describe will be used to generate a transcript for a video. Do not mention anything about tiktok. You can start by describing this image"},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Describe the image in an interesting and captivating manner that is short and succinct."},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{imageBase64}"}
                        },
                        

                    ]}
                ]
        if additionalContext:
            formattedMessages[1]['content'].append({"type": "text", "text": additionalContext})
        completion = gptClient.chat.completions.create(
            model='gpt-4o',
            messages=formattedMessages,
            temperature=0.7,
        )
    description = completion.choices[0].message.content
    print(description)
    
    return description

# Function to generate transcript with previous frame and evaluate the transcript using LLM as a judge
def getTranscriptv2(videoBinary, s3ObjectKey, username, additionalContext = None):
    
    # Smart Extract Keyframes
    keyframes,filtered_frames_indexes = extract_unique_scenes(videoBinary, autoFineTune=True)
    print(f'Extracted {len(keyframes)} keyframes')

    # Prereq 2 : Store to s3 Bucket 
    for idx, keyframeBinary in enumerate(keyframes):
        keyframepath = f"keyframes/{s3ObjectKey.split('/')[-1]}/{str(idx)}.jpg"
        s3Client.put_object(Bucket='videos-tiktok-backend', Key=keyframepath, Body=keyframeBinary)
        print(f"Uploaded keyframe {keyframes.index(keyframeBinary) + 1} to S3 successfully.")

    # 2) Describe Keyframes
    descriptions = []
    regeneration_Max_tries = 5
    for tries in range(regeneration_Max_tries):
        # Smarter Describe Keyframes
        for i, keyframe in enumerate(keyframes):
            if i == 0:
                description = describeImageGPT_v2(keyframe, additionalContext)
            else:
                description = describeImageGPT_v2(keyframe, descriptions[i-1], additionalContext)

            descriptions.append(description)
        transcript = '\n'.join(descriptions) 
        # Evaluate Transcript
        gptResponse = evaluate_transcript_gpt(transcript)
        # gptResponseImprovement = evaluate_transcript_gpt_v2(transcript)
        # additionalContext= gptResponseImprovement['improvement']
        geminiResponse = evaluate_transcript_gemini(transcript)
        # aggregate responses
        score = float((float(gptResponse) + float(geminiResponse)) / 2)
        print('Transcript Score:', score)
        if score > 7:
            break
        else:
            descriptions = []


    # Prereq 3 is to store individual keyframe descriptions to mongodb with associated s3ObjectKey
    for i in range(len(descriptions)):
        imageUrl = f"https://videos-tiktok-backend.s3.ap-southeast-1.amazonaws.com/keyframes/{s3ObjectKey.split('/')[-1]}/{str(i)}.jpg" 
        collection["keyframes"].insert_one({"imageUrl": imageUrl, "keyframeIndex": i, "description": descriptions[i], 'belongsTo' : username, "s3ObjectKey": s3ObjectKey, "filteredFrameIndexTimestamp": filtered_frames_indexes[i]})
        print(f"Inserted keyframe {i + 1} description to MongoDB successfully")
    
    # 3) Return Transcript
    transcript= '\n'.join(descriptions)
    

    return transcript

# Function to evaluate a TikTok transcript using the LLM GPT Endpoint
def evaluate_transcript_gpt(transcript):
    # Evaluate Transcript using GPT 3.5 Turbo
    rubrics = ['Interesting', 'Factually Possible', 'Succinct', 'Well-structured', 'Clear', 'Concise', 'Engaging']
    completion = gptClient.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
                {"role": "system", "content": f"You are an expert at evaluating TikTok transcripts for the visually impaired. Please evaluate the following transcript and provide a score from 1 to 10. You will only provide an integer as a response and nothing else. These are the criteria you will be evaluating the transcript on: {', '.join(rubrics)}."},
                {"role": "user", "content": [
                    {"type": "text", "text": f'Evaluate this transcript {transcript}'},
                ]}
            ],
            temperature=0.7,
        )
    score = completion.choices[0].message.content

    # Handle Parsing
    try:
        response = int(score)
    except ValueError:
        response = 0
        print("Error: Could not parse response as an integer")
    
    return response
# Function WIP to evaluate a TikTok transcript using the LLM GPT Endpoint and provide improvement suggestions
def evaluate_transcript_gpt_v2(transcript):
    # Model Setting
    rubrics = ['Interesting', 'Factually Possible', 'Succinct', 'Well-structured', 'Clear', 'Concise', 'Engaging']
    completion = gptClient.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
                {"role": "system", "content": f"You are an expert at evaluating TikTok transcripts for the visually impaired. Please evaluate the following transcript and provide a score from 1 to 10 and how to improve it. You will provide a output in a python dict of this exact format {{'score': '{{Your Score}}', 'improvement' : 'How the transcript can be improved' }}. These are the criteria you will be evaluating the transcript on: {', '.join(rubrics)}."},
                {"role": "user", "content": [
                    {"type": "text", "text": f'Evaluate this transcript {transcript}'},
                ]}
            ],
            temperature=0.7,
        )
    score = completion.choices[0].message.content

    # Handle Parsing
    try:
        response = dict(eval(score))
    except ValueError:
        response = 0
        print("Error: Could not parse response as an integer")
    
    return response
# Function to evaluate a TikTok transcript using the LLM Gemini Endpoint
def evaluate_transcript_gemini(transcript):
    rubrics = ['Interesting', 'Factually Possible', 'Succinct', 'Well-structured', 'Clear', 'Concise', 'Engaging']
    completion = geminiModel.generate_content(f"You are an expert at evaluating TikTok transcripts for the visually impaired. Please evaluate the following transcript and provide a score from 1 to 10. You will only provide an integer as a response and nothing else. These are the criteria you will be evaluating the transcript on: {', '.join(rubrics)}. Evaluate the following Transcript: {transcript}")
    response = completion.text
    try:
        response = int(response)
    except ValueError:
        response = 0
        print("Error: Could not parse response as an integer")
    return response

# endregion
if __name__ == "__main__":
    uvicorn.run(app)

