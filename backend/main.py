import os
import sys
sys.path.append(os.path.dirname(__file__))

import shutil
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response, Depends, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, EmailStr
from fpdf import FPDF
import io
from typing import List, Dict
import tempfile
from app_core import OpenAIClient, chat_graph, ChatState
from database import get_session_data, save_session_data, add_message_to_db, get_all_sessions, create_user, get_user_by_email
from auth import verify_password, get_password_hash, create_access_token, verify_token
import re

app = FastAPI(title="Medical Report Analyzer API")

# Setup static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "static")
if not os.path.exists(STATIC_DIR):
    # Fallback for Vercel flatten structures or different CWD
    STATIC_DIR = os.path.join(os.getcwd(), "frontend", "static")

# os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Auth Schemas & Dependency
class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

async def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    token = authorization.split(" ")[1]
    payload = verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Token expired or invalid")
    return payload.get("sub")

# Auth Endpoints
@app.post("/api/register")
async def register(user: UserCreate):
    # Strict Password Validation
    if len(user.password) < 8 or len(user.password) > 20:
        raise HTTPException(status_code=400, detail="Password must be between 8 and 20 characters")
    if not re.search(r'[A-Z]', user.password):
        raise HTTPException(status_code=400, detail="Password must contain at least one uppercase letter")
    if not re.search(r'\d', user.password):
        raise HTTPException(status_code=400, detail="Password must contain at least one number")

    existing = await get_user_by_email(user.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    
    hashed_pwd = get_password_hash(user.password)
    user_data = {
        "name": user.name,
        "email": user.email,
        "password": hashed_pwd
    }
    user_id = await create_user(user_data)
    token = create_access_token({"sub": user_id, "name": user.name})
    return {"access_token": token, "token_type": "bearer", "name": user.name}

@app.post("/api/login")
async def login(user: UserLogin):
    db_user = await get_user_by_email(user.email)
    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    if not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    token = create_access_token({"sub": str(db_user["_id"]), "name": db_user["name"]})
    return {"access_token": token, "token_type": "bearer", "name": db_user["name"]}


# Page Endpoints
@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open(os.path.join(STATIC_DIR, "index.html"), "r") as f:
        return f.read()

@app.get("/login", response_class=HTMLResponse)
async def read_login():
    with open(os.path.join(STATIC_DIR, "login.html"), "r") as f:
        return f.read()

@app.get("/register", response_class=HTMLResponse)
async def read_register():
    with open(os.path.join(STATIC_DIR, "register.html"), "r") as f:
        return f.read()

@app.get("/report-page/{session_id}", response_class=HTMLResponse)
async def read_report_page(session_id: str):
    with open(os.path.join(STATIC_DIR, "report.html"), "r") as f:
        return f.read()

# Secure API Endpoints
@app.get("/api/report/{session_id}")
async def get_report_api(session_id: str, user_id: str = Depends(get_current_user)):
    state = await get_session_data(session_id, user_id)
    if not state or not state["report_analysis"]:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return {"analysis": state["report_analysis"]}

@app.get("/api/sessions")
async def fetch_sessions(user_id: str = Depends(get_current_user)):
    sessions = await get_all_sessions(user_id)
    return {"sessions": sessions}

@app.post("/analyze")
async def analyze_report(file: UploadFile = File(...), session_id: str = Form("default"), user_id: str = Depends(get_current_user)):
    try:
        # Save uploaded file to temp
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        client = OpenAIClient()
        analysis = client.analyze_report(tmp_path)
        
        # Cleanup
        os.unlink(tmp_path)
        
        if analysis.startswith("Error"):
            return JSONResponse(content={"error": analysis}, status_code=400)
        
        state = await get_session_data(session_id, user_id)
        state["report_analysis"] = analysis
        state["is_report_analyzed"] = True
        
        await save_session_data(session_id, state, user_id)
        
        return {"analysis": analysis, "session_id": session_id}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/chat")
async def chat(message: str = Form(...), session_id: str = Form("default"), user_id: str = Depends(get_current_user)):
    state = await get_session_data(session_id, user_id)
    
    if not state["is_report_analyzed"]:
        return JSONResponse(content={"error": "Analyze a report first."}, status_code=400)
    
    state["current_query"] = message
    try:
        # LangGraph invoke is sync, but we are in an async route
        result = chat_graph.invoke(state)
        
        # Save messages to DB
        last_response = result["chat_history"][-1]["assistant"]
        await add_message_to_db(session_id, "user", message, user_id)
        await add_message_to_db(session_id, "assistant", last_response, user_id)
        await save_session_data(session_id, result, user_id)
        
        return {"response": last_response}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
