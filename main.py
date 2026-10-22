from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import chromadb
import os
import shutil
import json
import cv2

# --- NEW IMPORTS FOR .ENV & GEMINI API ---
from dotenv import load_dotenv
import google.generativeai as genai

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
    print("Gemini API Key successfully loaded from .env!")
else:
    print("Warning: GEMINI_API_KEY .env file mein nahi mili!")

# --- AI ERROR HELPER FUNCTION ---
def get_ai_help(error_message: str):
    if not api_key:
        return "AI Error Helper inactive hai kyunki GEMINI_API_KEY set nahi hai."
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = f"Yeh Python/FastAPI code mein error aaya hai, isko theek karne ka simple aur accurate solution batao: {error_message}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI se connect karne mein samasya aayi: {e}"

app = FastAPI(title="Satya-Net Multimodal RAG Engine", version="5.0", default_response_class=ORJSONResponse)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="fact_knowledge_base")

UPLOAD_FOLDER = "uploaded_images"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

AUDIO_UPLOAD_FOLDER = "uploaded_audio"
if not os.path.exists(AUDIO_UPLOAD_FOLDER):
    os.makedirs(AUDIO_UPLOAD_FOLDER)

users_db = {}
history_db = []
feedback_db = []

# --- TRUE LAZY LOADER FOR EMBEDDING MODEL ---
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("Loading Embedding Model (True Lazy)...")
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model

def load_facts_automatically():
    file_path = "facts.txt"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            
        model = get_embedding_model()
        for i, fact in enumerate(lines):
            embedding = model.encode(fact).tolist()
            try:
                collection.add(documents=[fact], embeddings=[embedding], ids=[str(i)])
            except Exception:
                pass 
        print(f"Successfully auto-loaded {len(lines)} facts from facts.txt into Vector Database!")
    else:
        print("Warning: facts.txt file not found!")

load_facts_automatically()

def analyze_image_visually(image_path):
    img = cv2.imread(image_path)
    if img is None: return "Error reading image."
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 100:
        return "Suspicious softness/artifact detected via Frequency Analysis."
    else:
        return "No obvious visual anomalies detected by basic analysis."

# --- TRUE LAZY LOADERS FOR DEEPFAKE MODELS ---
_deepfake_detector = None
_audio_detector = None

def get_deepfake_detector():
    global _deepfake_detector
    if _deepfake_detector is None:
        try:
            print("Loading Pre-trained Image Deepfake Model (True Lazy)...")
            from transformers import pipeline
            _deepfake_detector = pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection")
        except Exception as e:
            print(f"Error loading Image Deepfake Model: {e}")
    return _deepfake_detector

def get_audio_detector():
    global _audio_detector
    if _audio_detector is None:
        try:
            print("Loading Pre-trained Audio Deepfake Model (True Lazy)...")
            from transformers import pipeline
            _audio_detector = pipeline("audio-classification", model="MelodyMachine/Deepfake-audio-detection")
        except Exception as e:
            print(f"Error loading Audio Deepfake Model: {e}")
    return _audio_detector

# --- PYDANTIC MODELS ---
class ClaimRequest(BaseModel):
    text: str

class UserRegister(BaseModel):
    name: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class PasswordReset(BaseModel):
    email: str
    old_password: str
    new_password: str

class FeedbackRequest(BaseModel):
    name: str
    email: str
    message: str

@app.get("/api/status")
def home():
    return {"status": "Satya-Net Multimodal RAG Engine is Active & Ready!"}

@app.post("/api/register")
def register_user(user: UserRegister):
    if user.email in users_db:
        raise HTTPException(status_code=400, detail="User already registered with this email!")
    users_db[user.email] = {"name": user.name, "password": user.password}
    return {"status": "success", "message": "Account created successfully!"}

@app.post("/api/login")
def login_user(user: UserLogin):
    if user.email not in users_db or users_db[user.email]["password"] != user.password:
        raise HTTPException(status_code=401, detail="Invalid email or password!")
    return {"status": "success", "name": users_db[user.email]["name"], "message": "Login successful!"}

@app.post("/api/reset-password")
def reset_password(data: PasswordReset):
    if data.email not in users_db or users_db[data.email]["password"] != data.old_password:
        raise HTTPException(status_code=400, detail="Incorrect old password or email!")
    users_db[data.email]["password"] = data.new_password
    return {"status": "success", "message": "Password updated successfully!"}

@app.post("/verify-claim")
@app.post("/api/analyze")
def verify_claim(data: ClaimRequest):
    try:
        query_text = data.text.strip()
        if not query_text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if api_key:
            try:
                model = genai.GenerativeModel(
                    model_name="gemini-2.5-flash",
                    generation_config={"response_mime_type": "application/json"}
                )
                prompt = (
                    f"Aap ek expert fact-checker aur deepfake detector hain. Is claim ko analyze karein: '{query_text}'. "
                    f"Aapko strictly valid JSON format me response dena hai jisme yeh keys honi chahiye: "
                    f"'claim', 'verdict' (jaise 'False / Misleading (भ्रामक)', 'Likely Authentic / Verified', ya 'INCONCLUSIVE'), "
                    f"'confidence_score', aur 'explanation'."
                )
                response = model.generate_content(prompt)
                parsed_ai_data = json.loads(response.text)
                verdict = parsed_ai_data.get("verdict", "Analyzed by Gemini AI")
                explanation = parsed_ai_data.get("explanation", response.text)
            except Exception as ai_err:
                verdict = "AI Error"
                explanation = f"Gemini API call fail ho gayi: {ai_err}"
        else:
            verdict = "API Key Missing"
            explanation = "Gemini API key .env mein configured nahi hai."

        history_db.append({"type": "Text Claim", "query": query_text[:60] + "...", "verdict": verdict})
        return {"queried_text": query_text, "verdict": verdict, "explanation": explanation}
    except Exception as err:
        ai_fix = get_ai_help(str(err))
        raise HTTPException(status_code=500, detail=f"Internal Error: {err}. AI Suggestion: {ai_fix}")

@app.post("/verify-image")
def verify_image(image_file: UploadFile = File(...)):
    file_location = f"{UPLOAD_FOLDER}/{image_file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(image_file.file, buffer)
    
    opencv_status = analyze_image_visually(file_location)
    detector = get_deepfake_detector()
    
    deepfake_verdict = "Model not evaluated"
    confidence_score = 0.0
    
    if detector:
        try:
            results = detector(file_location)
            if results and len(results) > 0:
                top_result = results[0]
                deepfake_verdict = str(top_result.get('label', '')).lower()
                confidence_score = float(top_result.get('score', 0.0))
        except Exception:
            deepfake_verdict = "error"
            confidence_score = 0.0
    
    final_verdict = "INCONCLUSIVE"
    is_fake = any(keyword in deepfake_verdict for keyword in ["fake", "ai", "artificial", "generated", "synthetic"])
    is_real = any(keyword in deepfake_verdict for keyword in ["real", "authentic", "genuine"])

    if detector and "error" not in deepfake_verdict:
        if is_fake:
            final_verdict = "LIKELY AI-GENERATED / DEEPFAKE"
            final_explanation = f"Detected as Fake with {(confidence_score*100):.1f}% confidence. OpenCV Check: {opencv_status}"
        elif is_real:
            final_verdict = "LIKELY AUTHENTIC"
            final_explanation = f"Detected as Real with {(confidence_score*100):.1f}% confidence. OpenCV Check: {opencv_status}"
        else:
            if confidence_score > 0.4:
                final_verdict = f"MODEL VERDICT: {deepfake_verdict.upper()}"
                final_explanation = f"Confidence: {(confidence_score*100):.1f}%. OpenCV Check: {opencv_status}"
    else:
        final_verdict = "VISUAL ANALYSIS ONLY"
        final_explanation = f"Deepfake model bypassed. OpenCV Check: {opencv_status}"

    history_db.append({"type": "Image Forensics", "query": image_file.filename, "verdict": final_verdict})
    return {
        "filename": image_file.filename,
        "final_verdict": final_verdict,
        "final_explanation": final_explanation,
        "detection_layers": {
            "deepfake_model_output": {"raw_label": deepfake_verdict, "confidence": f"{(confidence_score*100):.1f}%"},
            "opencv_visual_check": opencv_status
        }
    }

@app.post("/verify-audio")
def verify_audio(audio_file: UploadFile = File(...)):
    file_location = f"{AUDIO_UPLOAD_FOLDER}/{audio_file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)
    
    detector = get_audio_detector()
    audio_verdict = "Model not evaluated"
    confidence_score = 0.0
    
    if detector:
        try:
            results = detector(file_location)
            if results and len(results) > 0:
                top_result = results[0]
                audio_verdict = str(top_result.get('label', '')).lower()
                confidence_score = float(top_result.get('score', 0.0))
        except Exception:
            audio_verdict = "error"
            confidence_score = 0.0
    
    final_verdict = "INCONCLUSIVE"
    is_fake_audio = any(kw in audio_verdict for kw in ["fake", "spoof", "ai", "generated", "clone"])
    is_real_audio = any(kw in audio_verdict for kw in ["real", "bonafide", "genuine"])

    if detector and "error" not in audio_verdict:
        if is_fake_audio:
            final_verdict = "LIKELY AI-GENERATED / CLONED VOICE"
            final_explanation = f"Detected as Spoof/Fake with {(confidence_score*100):.1f}% confidence."
        elif is_real_audio:
            final_verdict = "LIKELY AUTHENTIC VOICE"
            final_explanation = f"Detected as Genuine/Bonafide with {(confidence_score*100):.1f}% confidence."
        else:
            if confidence_score > 0.4:
                final_verdict = f"MODEL VERDICT: {audio_verdict.upper()}"
                final_explanation = f"Confidence: {(confidence_score*100):.1f}%"
    else:
        final_verdict = "AUDIO MODEL BYPASSED"
        final_explanation = "Audio classification model is not available."

    history_db.append({"type": "Audio Spoof", "query": audio_file.filename, "verdict": final_verdict})
    return {
        "filename": audio_file.filename,
        "final_verdict": final_verdict,
        "final_explanation": final_explanation,
        "audio_model_output": {"raw_label": audio_verdict, "confidence": f"{(confidence_score*100):.1f}%"}
    }

@app.get("/api/history")
def get_history():
    return {"history": history_db[::-1]}

@app.delete("/api/history/clear")
def clear_history():
    history_db.clear()
    return {"status": "success", "message": "History cleared successfully!"}

@app.post("/api/feedback")
def submit_feedback(feedback: FeedbackRequest):
    feedback_db.append(feedback.dict())
    return {"status": "success", "message": "Feedback received successfully. Thank you!"}

app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)