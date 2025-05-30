from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
import uvicorn
import os
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List
import json
import base64
import openai
from openai import OpenAI
from data.user_data import DEFAULT_USER_PROFILE, SUGGESTED_ACTIONS_DATA, DOCUMENTS_DATA

# Load environment variables
load_dotenv()

# OpenAI client initialization
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None

# Configuration - Update these paths according to your file locations
PDF_FILE_PATH = "document.pdf"     # Path to your PDF file

app = FastAPI(
    title="MayaCode Backend API",
    description="Backend API for MayaCode User Dashboard",
    version=os.getenv("API_VERSION", "v1")
)

# Pydantic models for chat
class ChatMessage(BaseModel):
    role: str  # "user", "assistant", or "system"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "gpt-3.5-turbo"  # Default model
    max_tokens: int = 150
    temperature: float = 0.7

class ChatResponse(BaseModel):
    response: str
    usage: Dict[str, Any] = None

# CORS configuration
origins = [
    "http://localhost:5173",  # Vite dev server
    "http://localhost:3000",  # Alternative React dev server
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "http://localhost:8000",  # Backend on localhost
    "http://127.0.0.1:8000",  # Backend on 127.0.0.1
    "*",  # Allow all origins for development
    os.getenv("CORS_ORIGIN", "http://localhost:5173")
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],  # Allow all headers
)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "success": True,
        "message": "MayaCode Backend API is running",
        "timestamp": datetime.now().isoformat(),
        "version": os.getenv("API_VERSION", "v1"),
        "pdf_exists": os.path.exists(PDF_FILE_PATH),
        "openai_configured": bool(openai_api_key)
    }

# Welcome route
@app.get("/")
async def welcome():
    return {"message": "FastAPI Backend is running!"}

@app.get("/user", response_model=Dict[str, Any])
async def get_user_data():
    try:
        # Combine all user data into a single response
        user_data = {
            **DEFAULT_USER_PROFILE,
            "suggestedActions": SUGGESTED_ACTIONS_DATA,
            "documents": DOCUMENTS_DATA
        }
        
        return {"success": True, "data": user_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading user data: {str(e)}")

@app.get("/pdf")
async def get_pdf_base64():
    try:
        if not os.path.exists(PDF_FILE_PATH):
            raise HTTPException(status_code=404, detail=f"PDF file '{PDF_FILE_PATH}' not found")
        with open(PDF_FILE_PATH, 'rb') as pdf_file:
            pdf_content = pdf_file.read()
            pdf_base64 = base64.b64encode(pdf_content).decode('utf-8')
        return {"success": True, "data": pdf_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF file: {str(e)}")

@app.post("/chat")
async def chat_with_openai(chat_request: ChatRequest):
    try:
        if not openai_api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        if not openai_client:
            raise HTTPException(status_code=500, detail="OpenAI client not initialized")
            
        messages = [{"role": msg.role, "content": msg.content} for msg in chat_request.messages]
        
        response = openai_client.chat.completions.create(
            model=chat_request.model,
            messages=messages,
            max_tokens=chat_request.max_tokens,
            temperature=chat_request.temperature
        )
        
        assistant_response = response.choices[0].message.content
        return {"response": assistant_response}
    except openai.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid OpenAI API key")
    except openai.RateLimitError:
        raise HTTPException(status_code=429, detail="OpenAI API rate limit exceeded")
    except openai.APIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)