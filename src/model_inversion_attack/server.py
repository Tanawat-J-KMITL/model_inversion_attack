import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from io import BytesIO
from PIL import Image
from model_inversion_attack import auth

app = FastAPI(title="Dummy Banking Biometric Auth Server")

BASE_URL = "/proxy/8000"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEBAPP_DIR = os.path.join(BASE_DIR, "..", "webapp")

# Simple in-memory session store (demo only)
sessions = {}

@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(os.path.join(WEBAPP_DIR, "index.html"))

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    session_id = request.cookies.get("session_id")

    if not session_id or session_id not in sessions:
        return RedirectResponse(url="/")

    return FileResponse(os.path.join(WEBAPP_DIR, "dashboard.html"))

@app.post("/api/v1/auth/{user_id}")
async def auth_endpoint(user_id: str, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    authenticated, confidence = auth.auth(image, user_id)
    response = JSONResponse({
        "authenticated": authenticated,
        "confidence": confidence
    })

    # If authenticated, create session
    if authenticated:
        session_id = user_id  # simple demo session
        sessions[session_id] = True
        response.set_cookie(key="session_id", value=session_id)

    return response

@app.get("/api/v1/logout")
async def logout(request: Request):
    session_id = request.cookies.get("session_id")
    if session_id in sessions:
        del sessions[session_id]

    response = RedirectResponse(url="/")
    response.delete_cookie("session_id")
    return response

def run_server(args):
    print("Starting Banking Biometric Auth Server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
