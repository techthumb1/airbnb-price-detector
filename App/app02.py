import random
import os
from fastapi import FastAPI
import numpy as np
from airbnb.abnb_bw import predict1


app = FastAPI(
    title='Data Science API project: Airbnb',
    description='By: Henry Gultom <br>Lambda School student',
    version='0.1',
    docs_url='/',
)

@app.get('/')
def root():
    """To get the price prediction, please scroll down and click the green box: **POST /predict** """
    return {"Hello" : "everyone"}


@app.post('/predict')
def predict(Property_type: str = 'Apartment', Room_type:str ='Private room',  Accommodates:int = 1, Bathrooms:float = 1.0, Bedrooms: int = 1, Beds:int=1, Bed_type: str= 'Real Bed', Cancellation_policy: str= 'flexible'):
    """ 
    1. click: **Try it out**  and change the value under the **Description**, and click: **Execute**.\n
    2. It predicts Airbnb rent price using XGBOOST as Machine Learning model.\n
    3. Scroll down to see the result at: **Responses**  >>  **Details**  >>  **Response body**
    """ 

    pred = predict1(Property_type, Room_type, Accommodates, Bathrooms, Bed_type, Cancellation_policy, Bedrooms, Beds)
    # pred = random.randint(1,200)
    
    res = {
            "predicted_price"   : round(float(pred),1)
        }

    return res
