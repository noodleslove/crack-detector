import os
import torch
import torchvision.transforms as transforms

from PIL import Image
from flask import Flask, redirect, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename

# Initialize the Flask app
app = Flask(__name__)

# Define maximum file size
app.config["MAX_CONTENT_LENGTH"] = 2 * 1000 * 1000

# Define the upload folder
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
UPLOAD_FOLDER = "uploads/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

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


def _allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            # Check if the post request has the file part
            return jsonify({"error": "no file"}), 400

        file = request.files["file"]
        # If the user does not select a file, the browser submits an
        # empty file without a filename
        if file.filename == "":
            return jsonify({"error": "empty filename"}), 400

        if file and _allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

            img = Image.open(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            input = transform(img).unsqueeze(0)

            # Make a prediction
            z = model(input)
            _, pred = torch.max(z, 1)

            # Return the prediction as JSON
            # return jsonify({"output": z.tolist(), "prediction": pred.item()}), 200
            return redirect(url_for("result", prediction=pred.item()))

        return jsonify({"error": "invalid file"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/result", methods=["GET"])
def result():
    pred = request.args.get("prediction")
    return render_template("index.html", prediction=pred)


if __name__ == "__main__":
    app.run(debug=True)
