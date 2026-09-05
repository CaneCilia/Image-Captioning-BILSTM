from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uuid

app = FastAPI(title=\"Image Captioning API\")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[\"*\"],
    allow_credentials=True,
    allow_methods=[\"*\"],
    allow_headers=[\"*\"],
)

UPLOAD_DIR = \"uploads\"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get(\"/\")
def read_root():
    return {\"message\": \"Welcome to the Image Captioning API\"}

@app.post(\"/upload-image/\")
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type.startswith(\"image/\"):
        raise HTTPException(status_code=400, detail=\"File provided is not an image.\")

    # Generate a unique filename
    file_extension = file.filename.split(\".\")[-1]
    unique_filename = f\"{uuid.uuid4()}.{file_extension}\"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    # Save the file
    with open(file_path, \"wb\") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {\"filename\": unique_filename, \"message\": \"Image uploaded successfully\", \"file_path\": file_path}
