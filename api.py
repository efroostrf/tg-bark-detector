from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from bson import ObjectId
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import uvicorn
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

# Load environment variables
load_dotenv()

# MongoDB connection
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'bark_system')

# Initialize FastAPI
app = FastAPI(title="Bark Detector API", description="API for dog bark detection system")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Custom JSON encoder to handle MongoDB ObjectId and datetime
class MongoJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, bytes):
            return "<binary data>"
        return super(MongoJSONEncoder, self).default(obj)

# Database connection
def get_db():
    client = MongoClient(MONGODB_URI)
    return client[MONGODB_DB_NAME]

@app.get("/")
async def root():
    return {"message": "Bark Detector API is running"}

@app.get("/noises")
async def get_noises(limit: int = 100, skip: int = 0):
    """
    Get a list of noise detections from the database, sorted from newest to oldest.
    """
    db = get_db()
    
    # Query the database
    cursor = db.noise_detections.find({}) \
                               .sort("timestamp", -1) \
                               .skip(skip) \
                               .limit(limit)
    
    # Convert to list and process each document
    result = []
    for doc in cursor:
        # Remove binary audio data from the response to reduce size
        if "audio_data" in doc:
            doc["has_audio"] = True
            del doc["audio_data"]
        else:
            doc["has_audio"] = False
            
        # Convert ObjectId to string for JSON serialization
        doc["_id"] = str(doc["_id"])
        
        result.append(doc)
    
    return result

@app.get("/noises/{noise_id}")
async def get_noise(noise_id: str):
    """
    Get a specific noise detection by ID.
    """
    try:
        db = get_db()
        
        # Convert string ID to ObjectId
        object_id = ObjectId(noise_id)
        
        # Find the document
        doc = db.noise_detections.find_one({"_id": object_id})
        
        if not doc:
            raise HTTPException(status_code=404, detail="Noise detection not found")
        
        # Remove binary audio data from the response
        if "audio_data" in doc:
            doc["has_audio"] = True
            del doc["audio_data"]
        else:
            doc["has_audio"] = False
            
        # Convert ObjectId to string
        doc["_id"] = str(doc["_id"])
        
        return doc
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=400, detail=f"Invalid ID format or other error: {str(e)}")

@app.get("/noises/{noise_id}/listen")
async def listen_noise(noise_id: str):
    """
    Stream the audio of a specific noise detection for direct playback in the browser.
    """
    try:
        db = get_db()
        
        # Convert string ID to ObjectId
        object_id = ObjectId(noise_id)
        
        # Find the document
        doc = db.noise_detections.find_one({"_id": object_id})
        
        if not doc:
            raise HTTPException(status_code=404, detail="Noise detection not found")
        
        # Check if audio data exists
        if "audio_data" not in doc or not doc["audio_data"]:
            raise HTTPException(status_code=404, detail="No audio data available for this detection")
        
        # Get audio format from document or default to wav
        audio_format = doc.get("audio_format", "wav")
        
        # Set appropriate content type for browser playback
        if audio_format == "wav":
            content_type = "audio/wav"
        elif audio_format == "ogg":
            content_type = "audio/ogg; codecs=opus"
        else:
            content_type = "application/octet-stream"
        
        # Return audio data with appropriate content type for inline playback
        # Note: Removing Content-Disposition header to prevent download
        return Response(
            content=doc["audio_data"],
            media_type=content_type
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=400, detail=f"Invalid ID format or other error: {str(e)}")

@app.get("/noises/{noise_id}/player", response_class=HTMLResponse)
async def noise_player(noise_id: str):
    """
    Provides a simple HTML page with an audio player for the noise.
    """
    try:
        db = get_db()
        
        # Convert string ID to ObjectId
        object_id = ObjectId(noise_id)
        
        # Find the document (just to check if it exists)
        doc = db.noise_detections.find_one({"_id": object_id})
        
        if not doc:
            raise HTTPException(status_code=404, detail="Noise detection not found")
        
        # Check if audio data exists
        if "audio_data" not in doc:
            return HTMLResponse(content="<html><body><h1>No audio available for this detection</h1></body></html>")
        
        # Get timestamp
        timestamp = doc.get("timestamp_str", "Unknown time")
        
        # Get classification info if available
        is_dog_bark = doc.get("is_dog_bark", False)
        classification_info = ""
        if is_dog_bark and "dog_classes" in doc:
            classification_info = f"<p>AI detected: {', '.join(doc['dog_classes'])}</p>"
        
        # Create a simple HTML page with an audio player
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Noise Recording Player</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }}
                .container {{ max-width: 800px; margin: 0 auto; }}
                h1 {{ color: #333; }}
                .player-card {{ 
                    background: #f5f5f5; 
                    border-radius: 8px; 
                    padding: 20px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
                }}
                audio {{ width: 100%; margin: 15px 0; }}
                .timestamp {{ color: #666; font-size: 0.9em; }}
                .dog-bark {{ 
                    color: {"#e53935" if is_dog_bark else "#666"}; 
                    font-weight: {"bold" if is_dog_bark else "normal"}; 
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Noise Recording Player</h1>
                <div class="player-card">
                    <p class="timestamp">Recorded at: {timestamp}</p>
                    <p class="dog-bark">{"🐶 DOG BARK DETECTED" if is_dog_bark else "No dog bark detected"}</p>
                    {classification_info}
                    <audio controls autoplay>
                        <source src="/noises/{noise_id}/listen" type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
                </div>
            </div>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html_content)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        return HTMLResponse(
            content=f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>",
            status_code=400
        )

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=3000, reload=True) 