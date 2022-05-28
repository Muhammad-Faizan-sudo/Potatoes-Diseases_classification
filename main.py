from distutils.log import debug
from io import BytesIO
from fastapi import FastAPI, UploadFile, File
from enum import Enum
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
from tensorflow.keras import layers, models


app = FastAPI()

MODEL = models.load_model("models/Potatoes_Disease_classfication_model.h5")

CLASSES = ['Potato___Early_blight', 
           'Potato___Late_blight', 
           'Potato___healthy'
          ]


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.get("/")
async def test():
    return "Welcome to Potatoes diseases classification model"


# @app.post("/predict")
# async def model_prediction(
#     file: bytes = File()
#     ):
#     # image = read_file_as_image(await file.read())
#     # img_batch = np.expand_dims(image, 0)
#     # y_predict = MODEL.predict(img_batch)
#     return {"Prediction":(file)}

@app.post("/predict/")
async def create_file(file: UploadFile = File(...)):
    img = read_file_as_image(await file.read())
    w,h,c = img.shape
    img = img.reshape(1,w,h,c)
    print(img)
    try:
        prediction = MODEL.predict(img)
        y_hat = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction, axis=1)[0]*100
        return {"Model Outcome": CLASSES[y_hat], "Confidence":confidence}
    except:
        return {"Model Outcome": None, "confidence":None}
    

if __name__ == "__main__":
    
    uvicorn.run(app, host='localhost', port=5000)

