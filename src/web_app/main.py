from fastapi import FastAPI, UploadFile, File
from keras import models
from keras import preprocessing
import numpy as np
import io
import uvicorn

app = FastAPI()

# Load your pre-trained model
model = models.load_model('path_to_your_model.h5')

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the image file
    contents = await file.read()
    img = preprocessing.image.load_img(io.BytesIO(contents), target_size=(224, 224))
    
    # Preprocess the image
    img_array = preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    
    return {"predicted_class": int(predicted_class[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)