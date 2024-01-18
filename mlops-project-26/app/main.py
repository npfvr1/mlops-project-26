import cv2
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

app = FastAPI()

# Load the EfficientNetB0 model with pre-trained ImageNet weights
model = EfficientNetB0(weights='imagenet')

@app.post("/predict/bird/")
async def predict_bird(file: UploadFile = File(...)):
    """Endpoint to predict the bird species in an image using EfficientNetB0."""
    try:
        # Read the uploaded image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image = image.resize((224, 224))
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)

        # Predict the class of the image
        predictions = model.predict(image_array)
        # Decode the predictions to get the class label
        decoded_predictions = decode_predictions(predictions, top=1)[0][0][1]  # Get the class name

        return JSONResponse(content={"class_name": decoded_predictions})

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})