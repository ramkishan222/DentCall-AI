from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import requests
import uvicorn
import os
import nest_asyncio
from openai import OpenAI
import httpx
import time

# Initialize FastAPI app
app = FastAPI()

# Telnyx and OpenAI API keys
TELNYX_API_KEY = os.getenv("TELNYX_API_KEY")  # Ensure these are set in the environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Store conversation history
conversation_history = [
    {"role": "system", "content": "Today you work at Telnyx. Sell Telnyx as best as you can."}
]

# Initialize nest_asyncio
nest_asyncio.apply()

# Endpoint to handle incoming Webhook events
@app.post("/webhook")
async def telnyx_webhook(request: Request):
    try:
        # Await the json method to parse the incoming JSON data
        incoming_data = await request.json()
        
        # Extract `event_type` and validate its presence
        event_type = incoming_data.get('data', {}).get('event_type')
        if not event_type:
            return JSONResponse(
                content={"status": "error", "message": "'event_type' key is missing"},
                status_code=400
            )

        if event_type == 'call.initiated':
            return answer_call(incoming_data['data']['payload']['call_control_id'])
        elif event_type == 'call.answered':
            return start_transcription(incoming_data['data']['payload']['call_control_id'])
        elif event_type == 'call.transcription':
            return handle_transcription(incoming_data['data']['payload']['call_control_id'], incoming_data['data']['payload']['transcription_data']['transcript'])
        
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )

# Function to handle call initiation
def answer_call(call_control_id):
    url = f"https://api.telnyx.com/v2/calls/{call_control_id}/actions/answer"
    headers = {"Authorization": f"Bearer {TELNYX_API_KEY}"}
    response = requests.post(url, headers=headers)
    return JSONResponse({"status": "answered", "Message": response.text})

# Function to start transcription
def start_transcription(call_control_id):
    url = f"https://api.telnyx.com/v2/calls/{call_control_id}/actions/transcription_start"
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f"Bearer {TELNYX_API_KEY}"
    }
    
    data = {
        "language": "en-US",
        "transcription_engine": "B",
        "transcription_tracks": "inbound"
    }
    
    response = requests.post(url, json=data, headers=headers)
    return JSONResponse({"status": "transcription_started", "Message": response.text})

# Pause transcription function
def stop_transcription(call_control_id):
    url = f"https://api.telnyx.com/v2/calls/{call_control_id}/actions/transcription_stop"
    headers = {"Authorization": f"Bearer {TELNYX_API_KEY}"}
    response = requests.post(url, headers=headers)
    print("Transcription paused.")
    return response.text

# Resume transcription function
def resume_transcription(call_control_id):
    url = f"https://api.telnyx.com/v2/calls/{call_control_id}/actions/transcription_start"
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f"Bearer {TELNYX_API_KEY}"
    }
    
    data = {
        "language": "en-US",
        "transcription_engine": "B",
        "transcription_tracks": "inbound"
    }
    
    response = requests.post(url, json=data, headers=headers)
    print("Transcription resumed.")
    return response.text

# Function to handle transcription and interruption
def handle_transcription(call_control_id, transcript):
    if not transcript:
        return JSONResponse({"status": "no_speech_detected", "message": "No speech detected, waiting for input."})
    
    # Handle interruption and resumption logic
    if "pause" in transcript.lower() or "stop" in transcript.lower():
        stop_transcription(call_control_id)  # Pause transcription if interruption occurs
        return JSONResponse({"status": "paused", "message": "Transcription paused due to interruption."})

    # If no interruption, process the transcription as usual
    url = "http://127.0.0.1:8080/v1/msg-servive"
    headers = {
        "X-API-KEY": "your-api-key-here", 
        "Content-Type": "application/json",
    }
    data = {
        "human": transcript,
    }
    
    try:
        response = httpx.post(url, headers=headers, json=data)
        response.raise_for_status()

        # Parse the response JSON
        response_data = response.json()
        model_response = response_data.get('model_response', 'No response available')
        
        # Send the response back to Telnyx as TTS
        return send_tts_response(call_control_id, model_response)
    except httpx.HTTPError as e:
        print(f"Failed to initiate API: {e}")
        return None

# Function to send TTS response to Telnyx
def send_tts_response(call_control_id, text):
    url = f"https://api.telnyx.com/v2/calls/{call_control_id}/actions/speak"
    headers = {"Authorization": f"Bearer {TELNYX_API_KEY}"}
    data = {"payload": text, "stop": "current", "voice": "female", "language": "en-US"}
    
    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code == 200:
        return JSONResponse({"status": "greeting_sent", "Message": response.text})
    else:
        return JSONResponse({"status": "failed", "error": response.text})

# Function to send greeting (initial greeting)
def send_greeting(call_control_id):
    url = f"https://api.telnyx.com/v2/calls/{call_control_id}/actions/speak"
    headers = {"Authorization": f"Bearer {TELNYX_API_KEY}"}
    data = {"payload": "Hello! Thank you for calling. How can we assist you today?", "stop": "current", "voice": "female", "language": "en-US"}
    
    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code == 200:
        return JSONResponse({"status": "greeting_sent", "Message": response.text})
    else:
        return JSONResponse({"status": "failed", "error": response.text})

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)