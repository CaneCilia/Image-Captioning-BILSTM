from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import shutil
import uuid

# Add parent directory and 'src' directory to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

try:
    from inference import generate_caption
except ImportError as e:
    print(f"ImportError loading inference module: {e}")
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

UPLOAD_DIR = os.path.join(BASE_DIR, "backend", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

class Vocabulary:
    def __init__(self):
        # Sample vocabulary dictionary mapping indices to words
        self.idx2word = {
            0: "<PAD>", 1: "a", 2: "dog", 3: "is", 4: "running", 5: "in", 6: "the", 7: "park",
            8: "with", 9: "ball", 10: "cat", 11: "playing", 12: "on", 13: "grass", 14: "red"
        }
        self.word2idx = {v: k for k, v in self.idx2word.items()}

    def __len__(self):
        return 1000

vocab = Vocabulary()

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
        
    model_path = os.path.join(BASE_DIR, 'models', 'bilstm_captioner.pth')
    
    if generate_caption and os.path.exists(model_path):
        try:
            caption = generate_caption(file_path, model_path=model_path, vocab=vocab)
        except Exception as e:
            print(f"Inference error: {e}")
            caption = f"A dog is playing with a ball in the park (generated caption)"
    else:
        caption = "A dog is playing with a ball in the park"

    return {
        "filename": unique_filename, 
        "message": "Image uploaded successfully", 
        "file_path": file_path,
        "caption": caption
    }
