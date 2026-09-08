from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import shutil
import uuid

# Add the parent directory (which contains 'src') to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src.inference import generate_caption
except ImportError:
    # Fallback if inference.py isn't fully implemented yet
    generate_caption = None

app = FastAPI(title="Image Captioning API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Image Captioning API"}

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    # Generate a unique filename
    file_extension = file.filename.split(".")[-1]
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    # Save the file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # If the model is fully implemented, we would run inference here
    # For now, we will simulate the caption generation
    caption = "Placeholder generated caption"
    
    if generate_caption:
        try:
            class MockVocab:
                def __init__(self):
                    self.idx2word = {1: "a", 2: "cute", 3: "dog", 4: "is", 5: "playing", 6: "with", 7: "a", 8: "ball"}
                def __len__(self):
                    return 1000
            vocab = MockVocab()
            
            # Create a mock features method for YOLOFeatureExtractor because it was crashing with cv2 missing etc
            # But wait, inference.py uses YOLOFeatureExtractor. 
            caption = generate_caption(file_path, model_path='../models/bilstm_captioner.pth', vocab=vocab)
        except Exception as e:
            print(f"Inference error: {e}")
            caption = f"Inference failed: {e}"

    return {
        "filename": unique_filename, 
        "message": "Image uploaded successfully", 
        "file_path": file_path,
        "caption": caption
    }
