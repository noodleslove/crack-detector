import os
import torch
import torchvision.transforms as transforms

from PIL import Image
from fastapi import FastAPI, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from werkzeug.utils import secure_filename

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define maximum file size
MAX_CONTENT_LENGTH = 2 * 1000 * 1000

# Define the upload folder
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
UPLOAD_FOLDER = "uploads/"

# Create the uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the pre-trained model
model = torch.load("model.pt")
model.eval()

# Define the input transformation
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _allowed_file(file):
    """Check if the file is allowed based on the extension and size."""
    return (
        "." in file.filename
        and file.filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
        and file.size <= MAX_CONTENT_LENGTH
    )
    # End of function


def _save_file(file) -> str:
    """Save the uploaded file to the upload folder."""
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    return filename
    # End of function


@app.post("/predict", status_code=200)
async def predict(file: UploadFile, response: Response):
    try:
        if file.filename == "":
            response.status_code = 400
            return {"error": "empty filename"}

        if file and _allowed_file(file):
            filename = _save_file(file)

            img = Image.open(os.path.join(UPLOAD_FOLDER, filename))
            input = transform(img).unsqueeze(0)

            # Make a prediction
            z = model(input)
            _, pred = torch.max(z, 1)

            # Return the prediction as JSON
            return {"output": z.tolist(), "prediction": pred.item()}

        response.status_code = 400
        return {"error": "invalid file"}
        # End of try block
    except Exception as e:
        response.status_code = 500
        return {"error": str(e)}
        # End of except block
