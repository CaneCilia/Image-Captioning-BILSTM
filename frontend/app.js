document.getElementById('uploadBtn').addEventListener('click', async () => {
    const fileInput = document.getElementById('imageInput');
    const resultSection = document.getElementById('resultSection');
    const previewImage = document.getElementById('previewImage');
    const captionText = document.getElementById('captionText');

    if (fileInput.files.length === 0) {
        alert('Please select an image first.');
        return;
    }

    const file = fileInput.files[0];
    
    // Show preview
    previewImage.src = URL.createObjectURL(file);
    resultSection.style.display = 'block';
    captionText.innerText = 'Generating caption...';

    // Prepare FormData
    const formData = new FormData();
    formData.append('file', file);

    try {
        // Send to FastAPI backend
        const response = await fetch('http://127.0.0.1:8000/upload-image/', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.statusText}`);
        }

        const data = await response.json();
        
        // Display result
        captionText.innerText = data.caption || 'Caption generation failed.';
        
    } catch (error) {
        console.error('Error:', error);
        captionText.innerText = 'Error generating caption. Is the backend running?';
    }
});

document.getElementById('imageInput').addEventListener('change', function() {
    const fileName = this.files.length > 0 ? this.files[0].name : "Click here to Select Image !!";
    document.querySelector('.custom-file-upload').textContent = fileName;
});
