from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
# import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter
import json
import io
import os

app = FastAPI(title="Crop Disease Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model_float32.tflite")
LABELS_PATH = os.path.join(BASE_DIR, "model", "class_labels.json")

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(LABELS_PATH) as f:
    class_names = json.load(f)

GATE_MODEL_PATH = os.path.join(BASE_DIR, "model", "model_gate_float32.tflite")
GATE_LABELS_PATH = os.path.join(BASE_DIR, "model", "gate_class_labels.json")

gate_interpreter = Interpreter(model_path=GATE_MODEL_PATH)
gate_interpreter.allocate_tensors()
gate_input_details = gate_interpreter.get_input_details()
gate_output_details = gate_interpreter.get_output_details()

with open(GATE_LABELS_PATH) as f:
    gate_class_names = json.load(f) 

IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.expand_dims(arr, axis=0)  
    return arr.astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits))
    return exp / exp.sum()


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Crop Disease Detection API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_tensor = preprocess_image(image_bytes)

    # Stage A: Gate check — leaf naki not_leaf
    gate_interpreter.set_tensor(gate_input_details[0]["index"], input_tensor)
    gate_interpreter.invoke()
    gate_output = gate_interpreter.get_tensor(gate_output_details[0]["index"])[0]
    gate_probs = softmax(gate_output)
    gate_predicted_idx = int(np.argmax(gate_probs))
    gate_predicted_class = gate_class_names[gate_predicted_idx]
    gate_confidence = float(gate_probs[gate_predicted_idx])

    if gate_predicted_class == "not_leaf":
        return {
            "disease": None,
            "confidence": round(gate_confidence, 4),
            "message": "This doesn't look like a Corn, Potato, or Tomato leaf. Please take a clear photo of one of these crops.",
            "all_probabilities": {},
        }

    # Stage B: leaf হলে, আসল diagnosis model চালাও (অপরিবর্তিত, ৯৬.৪০% accuracy)
    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]

    probabilities = softmax(output)
    predicted_idx = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_idx]
    confidence = float(probabilities[predicted_idx])

    return {
        "disease": predicted_class,
        "confidence": round(confidence, 4),
        "message": None,
        "all_probabilities": {
            class_names[i]: round(float(p), 4) for i, p in enumerate(probabilities)
        },
    }

@app.get("/supported-diseases")
def supported_diseases():
    return {
        "num_classes": len(class_names),
        "classes": class_names,
        "note": "This model only detects diseases for Corn, Potato, and Tomato leaves."
    }