import os
import io
import sys
import cv2
import base64
import requests
import asyncio
import logging
import numpy as np
from pathlib import Path
from PIL import Image
from flask import Flask, request, jsonify, render_template
from werkzeug.datastructures import FileStorage
from dotenv import load_dotenv
from deepface import DeepFace
from typing import Union, Tuple
from concurrent.futures import ThreadPoolExecutor

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 5000))
DEBUG = os.getenv("DEBUG", "False") == "True"
APP_NAME = os.getenv("APP_NAME", "")
APP_VERSI = os.getenv("APP_VERSI", "")
DEV_NAME = os.getenv("DEV_NAME", "")
DEV_EMAIL = os.getenv("DEV_EMAIL", "")

# DeepFace models and backends configuration
models = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "Dlib",
    "SFace",
    "GhostFaceNet",
]
backends = [
    "opencv",
    "ssd",
    "dlib",
    "mtcnn",
    "fastmtcnn",
    "retinaface",
    "mediapipe",
    "yolov8",
    "yunet",
    "centerface",
]

# Set logging level
log_level = logging.DEBUG if DEBUG else logging.ERROR
logging.basicConfig(
    filename="error.log",
    level=log_level,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ThreadPoolExecutor to handle blocking calls
executor = ThreadPoolExecutor(max_workers=2)


async def load_image_from_file_storage(file: FileStorage) -> np.ndarray:
    """Decode image from FileStorage."""
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    return img


async def load_image_from_base64(uri: str) -> np.ndarray:
    """Decode image from base64 encoded string."""
    encoded_data = uri.split(",")[1]
    decoded_bytes = base64.b64decode(encoded_data)

    with Image.open(io.BytesIO(decoded_bytes)) as img:
        file_type = str(img.format).lower()
        if file_type not in {"jpeg", "png"}:
            raise ValueError(f"Image must be jpg or png, but received {file_type}")

    img_array = np.frombuffer(decoded_bytes, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


async def load_image_from_web(url: str) -> np.ndarray:
    """Load and decode image from a URL."""
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    img_array = np.asarray(bytearray(response.raw.read()), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


async def load_image(img_path: Union[str, np.ndarray]) -> Tuple[np.ndarray, str]:
    """Load image from file, URL, base64, or numpy array."""
    if isinstance(img_path, np.ndarray):
        return img_path, "numpy array"

    if isinstance(img_path, Path):
        img_path = str(img_path)

    if isinstance(img_path, str):
        if img_path.startswith("data:image/"):
            return await load_image_from_base64(img_path), "base64"
        if img_path.lower().startswith(("http://", "https://")):
            return await load_image_from_web(img_path), img_path
        if not os.path.isfile(img_path):
            raise ValueError(f"Image file does not exist: {img_path}")
        return cv2.imread(img_path), img_path


async def extract_image_from_request(img_key: str) -> np.ndarray:
    """Extract image from POST request (file or base64)."""
    if request.files:
        file = request.files.get(img_key)
        if not file:
            raise ValueError(f"Missing image file for key '{img_key}'")
        return await load_image_from_file_storage(file)

    input_args = request.get_json() or request.form.to_dict()
    img_data = input_args.get(img_key)
    if not img_data:
        raise ValueError(f"Missing image data for key '{img_key}'")

    img, _ = await load_image(img_data)
    return img


def verify_faces_sync(img1: np.ndarray, img2: np.ndarray) -> dict:
    """Verify faces between two images."""
    return DeepFace.verify(
        img1, img2, model_name=models[1], detector_backend=backends[1], threshold=0.5
    )


async def verify_faces(img1: np.ndarray, img2: np.ndarray) -> dict:
    """Run DeepFace face verification asynchronously in a thread."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, verify_faces_sync, img1, img2)


def extract_faces_sync(img: np.ndarray) -> list:
    """Extract faces from an image."""
    faces = DeepFace.extract_faces(img, detector_backend=backends[1])
    result = [
        {"facial_area": face["facial_area"], "confidence": face["confidence"]}
        for face in faces
    ]
    return result


async def extract_faces(img: np.ndarray) -> list:
    """Run DeepFace face extraction asynchronously in a thread."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, extract_faces_sync, img)


def represent_faces_sync(img: np.ndarray) -> list:
    """Represent faces from an image."""
    return DeepFace.represent(img, model_name=models[1], detector_backend=backends[1])


async def represent_faces(img: np.ndarray) -> list:
    """Run DeepFace face representation asynchronously in a thread."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, represent_faces_sync, img)


# Route to render home page
@app.route("/")
async def home():
    return render_template(
        "home.html", title=APP_NAME, app_name=APP_NAME, app_versi=APP_VERSI
    )


# Error handler for 404
@app.errorhandler(404)
async def page_not_found(error):
    return render_template(
        "404.html", title="404 - Page Not Found", app_name=APP_NAME, app_versi=APP_VERSI
    )


@app.before_request
async def check_token():
    """Check if the request has a valid authorization token."""
    if request.method.upper() in ["OPTIONS", "GET"]:
        return

    token = request.headers.get("Authorization")
    if not token or token.split()[1] != os.getenv("ACCESS_TOKEN"):
        return (
            jsonify({"ok": False, "code": 401, "message": "Invalid or missing token"}),
            401,
        )


@app.route("/verify", methods=["POST"])
async def verify():
    """API endpoint for face verification."""
    try:
        img1 = await extract_image_from_request("img1")
        img2 = await extract_image_from_request("img2")
        result = await verify_faces(img1, img2)
        return jsonify({"ok": True, "code": 200, "data": result}), 200
    except Exception as error:
        return (
            jsonify(
                {
                    "ok": False,
                    "code": 400,
                    "message": f"Error: {error}",
                }
            ),
            400,
        )


@app.route("/extract", methods=["POST"])
async def extract():
    """API endpoint for face extraction."""
    try:
        img = await extract_image_from_request("img")
        result = await extract_faces(img)
        return jsonify({"ok": True, "code": 200, "data": result}), 200
    except Exception as error:
        return (
            jsonify(
                {
                    "ok": False,
                    "code": 400,
                    "message": f"Error: {error}",
                }
            ),
            400,
        )


@app.route("/represent", methods=["POST"])
async def represent():
    """API endpoint for face representation."""
    try:
        img = await extract_image_from_request("img")
        result = await represent_faces(img)
        return jsonify({"ok": True, "code": 200, "data": result}), 200
    except Exception as error:
        return (
            jsonify(
                {
                    "ok": False,
                    "code": 400,
                    "message": f"Error: {error}",
                }
            ),
            400,
        )


if __name__ == "__main__":
    try:
        app.run(host=HOST, port=PORT, debug=DEBUG)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
