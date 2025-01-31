from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import speech_recognition as sr
import kokoro
from kokoro import KPipeline
import os
import tempfile
import datetime
import base64
import io
import subprocess
from pydub import AudioSegment
from pydub.generators import Sine
import openai
from pydantic import BaseModel
import uvicorn

# Ensure the ffmpeg path is set correctly
os.environ["PATH"] += os.pathsep + "C:\\ffmpeg\\bin"

app = FastAPI()

# Mount the responses directory to make files accessible
app.mount("/responses", StaticFiles(directory="responses"), name="responses")

# Initialize the speech recognizer and text-to-speech engine
recognizer = sr.Recognizer()
tts_engine = KPipeline(model_name="kokoro-82m")  # Corrected KokoroTTS initialization

# Configure OpenAI (replace with your API key)
openai.api_key = "your-api-key"

# Initialize OpenAI client
client = openai.OpenAI(api_key="your-api-key")

# Optimize prompt for voice conversation context
SYSTEM_PROMPT = """You are a voice assistant in a real-time conversation. 
Keep responses concise and natural, as if speaking.
Maintain context but be brief, as this is a voice call."""

class Conversation:
    def __init__(self):
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

conversation = Conversation()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.listening_status: dict[WebSocket, bool] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.listening_status[websocket] = False

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        if websocket in self.listening_status:
            del self.listening_status[websocket]

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_json({
            "type": "message",
            "data": message
        })

    async def send_audio(self, audio_data: bytes, websocket: WebSocket):
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        await websocket.send_json({
            "type": "audio",
            "data": audio_base64
        })

manager = ConnectionManager()

# Add this function to find ffmpeg executable
def find_ffmpeg():
    try:
        # Try to find ffmpeg in common locations
        common_paths = [
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
            r"C:\ffmpeg\bin\ffmpeg.exe",
        ]
        
        # First check if ffmpeg is in PATH
        result = subprocess.run(['where', 'ffmpeg'], capture_output=True, text=True)
        if result.returncode == 0:
            ffmpeg_path = result.stdout.strip().split('\n')[0]
            return os.path.dirname(ffmpeg_path)
        
        # Check common locations
        for path in common_paths:
            if os.path.exists(path):
                return os.path.dirname(path)
        
        return None
    except Exception as e:
        print(f"Error finding ffmpeg: {e}")
        return None

# Set up ffmpeg paths
ffmpeg_path = find_ffmpeg()
if ffmpeg_path:
    print(f"Found ffmpeg at: {ffmpeg_path}")
    AudioSegment.converter = os.path.join(ffmpeg_path, "ffmpeg.exe")
    AudioSegment.ffmpeg = os.path.join(ffmpeg_path, "ffmpeg.exe")
    AudioSegment.ffprobe = os.path.join(ffmpeg_path, "ffprobe.exe")
else:
    print("Warning: ffmpeg not found. Please install ffmpeg")

# Add these constants for audio processing
NOISE_THRESHOLD = 0.015  # Adjust based on testing
MIN_AUDIO_LENGTH = 0.5  # Minimum audio length in seconds
MAX_SILENCE = 1.0  # Maximum silence duration in seconds

async def speech_to_text(audio_data):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio_path = temp_audio.name
            temp_audio.write(audio_data)
            temp_audio.flush()
            
            try:
                # Convert and analyze audio
                audio_segment = AudioSegment.from_file(temp_audio_path)
                
                # Check audio quality
                if len(audio_segment) < MIN_AUDIO_LENGTH * 1000:  # Convert to milliseconds
                    return "too_short"
                
                # Calculate RMS (Root Mean Square) to check audio level
                rms = audio_segment.rms
                if rms < NOISE_THRESHOLD * 32767:  # 32767 is max value for 16-bit audio
                    return "too_quiet"

                # Proceed with speech recognition
                converted_path = temp_audio_path + "_converted.wav"
                audio_segment.export(converted_path, format="wav")

                with sr.AudioFile(converted_path) as source:
                    audio = recognizer.record(source)
                    text = recognizer.recognize_google(audio)
                    return text if text.strip() else "no_speech"
                    
            except sr.UnknownValueError:
                return "no_speech"
            except sr.RequestError:
                return "error"
            finally:
                pass
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return "error"

def get_ai_response(text):
    conversation.messages.append({"role": "user", "content": text})
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # You can use a smaller model for cost optimization
        messages=conversation.messages,
        max_tokens=100,  # Limit response length to reduce costs
        temperature=0.7
    )
    
    ai_response = response.choices[0].message.content
    conversation.messages.append({"role": "assistant", "content": ai_response})
    return ai_response

def text_to_speech(text):
    temp_audio_path = None
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio_path = temp_audio.name

        # Generate audio using Kokoro
        audio_data = tts_engine.generate(text)
        
        # Save the audio data
        with open(temp_audio_path, "wb") as f:
            f.write(audio_data)

        # Read the audio data back
        with open(temp_audio_path, "rb") as audio_file:
            return audio_file.read()
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except PermissionError:
                pass

def save_audio_response(audio_data, text):
    # Create a filename based on the first few words of the text (sanitized)
    safe_text = "".join(x for x in text[:30] if x.isalnum() or x.isspace()).strip()
    safe_text = safe_text.replace(" ", "_")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"response_{safe_text}_{timestamp}.mp3"
    
    # Create responses directory if it doesn't exist
    os.makedirs("responses", exist_ok=True)
    
    # Save the audio file
    file_path = os.path.join("responses", filename)
    with open(file_path, "wb") as f:
        f.write(audio_data)
    return file_path

@app.post("/chat")
async def chat_endpoint(audio: UploadFile = File(...)):
    try:
        # Convert incoming audio to text
        audio_content = await audio.read()
        text = await speech_to_text(audio_content)
        
        # Get AI response
        ai_response = get_ai_response(text)
        
        # Convert AI response to speech
        audio_response = text_to_speech(ai_response)
        
        # Save the audio response
        saved_file_path = save_audio_response(audio_response, ai_response)
        print(f"Response saved to: {saved_file_path}")
        
        # Create download URL
        filename = os.path.basename(saved_file_path)
        download_url = f"/responses/{filename}"
        
        # Return audio stream
        return StreamingResponse(
            io.BytesIO(audio_response),
            media_type="audio/mpeg",
            headers={
                "X-Response-File": saved_file_path,
                "X-Response-Text": ai_response,
                "X-Download-URL": download_url
            }
        )
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_audio(filename: str):
    file_path = os.path.join("responses", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        file_path,
        media_type="audio/mpeg",
        filename=filename
    )

@app.get("/")
async def root():
    return {"message": "Voice-to-Voice Chatbot API is running"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    print("New client connected")
    
    try:
        while True:
            try:
                # Receive the audio data
                data = await websocket.receive_bytes()
                print(f"Received audio data: {len(data)} bytes")
                
                # Convert speech to text
                text = await speech_to_text(data)
                print(f"Converted text: {text}")
                
                # Only process valid speech
                if text and text not in ["too_short", "too_quiet", "no_speech", "error"]:
                    # Get AI response
                    ai_response = get_ai_response(text)
                    print(f"AI response: {ai_response}")
                    
                    # Send text response
                    await manager.send_message(ai_response, websocket)
                    
                    # Convert to speech and send audio
                    audio_response = text_to_speech(ai_response)
                    await manager.send_audio(audio_response, websocket)
                else:
                    print(f"Invalid speech detected: {text}")
                
            except WebSocketDisconnect:
                manager.disconnect(websocket)
                break
            except Exception as e:
                print(f"Error in websocket: {str(e)}")
                await manager.send_message("An error occurred. Please try again.", websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(
        "script:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        ws_max_size=1024*1024*10  # 10MB max message size
    )
