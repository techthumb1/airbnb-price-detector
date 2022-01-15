
import pandas as pd
import numpy as np

abnb = pd.read_csv('airbnb/Airbnb_U4.csv',
                   usecols=[1, 2, 3, 5, 6, 7, 8, 27, 28],
                   )
print(abnb.shape)

# abnb.info()

# print("\nNULL :\n", abnb.isnull().sum())

abnb['price'] = round(np.exp(abnb['log_price']),1)

print(abnb.dtypes)

# Remove rows with NULLS
abnb = abnb.dropna(axis = 0, how ='any')


# Convert bedrooms & beds to integer
abnb['bedrooms'] = abnb['bedrooms'].astype(int)
abnb['beds'] = abnb['beds'].astype(int)


# Drop certain criteria: zero beds or price, excessive price, etc...
abnb.drop(abnb[ abnb['price'] < 20 ].index , inplace=True)
abnb.drop(abnb[ abnb['price'] > 1500 ].index , inplace=True)
abnb.drop(abnb[ abnb['beds'] == 0 ].index , inplace=True)
abnb.drop(abnb[ abnb['bedrooms'] == 0 ].index , inplace=True)


# MACHINE LEARNING
# Define X & y

X_train = abnb.drop(columns=['log_price', 'price'])
y_train = abnb['price']



# Split into train & test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape,  X_test.shape, y_test.shape)

print('after -1')

# Use Neural Network

import category_encoders as ce
from sklearn.pipeline import make_pipeline

encoder = ce.OrdinalEncoder()
X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)


from keras import models, layers, optimizers, regularizers

# Building the model
nn2 = models.Sequential()
nn2.add(layers.Dense(64, input_shape=(X_train.shape[1],), activation='relu'))
nn2.add(layers.Dense(64, activation='relu'))
nn2.add(layers.Dense(64, activation='relu'))
nn2.add(layers.Dense(1, activation='linear')
) 

# Compiling the model
nn2.compile(loss='mse',
            optimizer='adam',    # or RMSprop , Adam
            metrics=['mean_absolute_error'])



# FITTING 
history = nn2.fit(X_train_encoded, y_train, 
                  epochs= 30,
                  batch_size=128,
                  validation_data=(X_test_encoded, y_test))


# predicted value
y_pred = nn2.predict(X_train_encoded)


# y_pred
y_pred = nn2.predict(X_test_encoded)
y_pred


# See the error: mse, rmse, R^2
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mse_nn2 = np.round(mean_squared_error(y_test, y_pred),3)
rmse_nn2 = np.round(np.sqrt(mse_nn2),3)
mae_nn2 = np.round(mean_absolute_error(y_test, y_pred),3)
r2_nn2 = np.round(r2_score(y_test, y_pred),3)

print('MAE   :', mae_nn2)
print('MSE   :', mse_nn2)
print('RMSE  :', rmse_nn2)
print('R^2 = :', r2_nn2)


# Price prediction based on single row inputs ..........
import shap 
encoder = ce.OrdinalEncoder()

# Using the predict function
def predict2(Property_type, Room_type, Accommodates, Bathrooms, Bed_type, Cancellation_policy, Bedrooms, Beds):

    # Make dataframe from the inputs
    dshap = pd.DataFrame(
        data=[[Property_type, Room_type, Accommodates, Bathrooms, Bed_type, Cancellation_policy, Bedrooms, Beds]], 
        columns=['property_type',	'room_type',	'accommodates',	'bathrooms',	'bed_type',	'cancellation_policy',	'bedrooms',	'beds']
    )

    dshap_encoded = encoder.fit_transform(dshap)


    # Get the model's prediction
    pred2 = nn2.predict(dshap_encoded)[0]
    result = round(float(pred2),2)
    print("=  $", result)
    return pred2


# Give the features as input and show the price:
Property_type = 'Apartment'
Room_type = 'Private room'
Accommodates = 1
Bathrooms = 1.0
Bedrooms = 1
Beds = 1
Bed_type = 'Real Bed'
Cancellation_policy = 'flexible'


print("\nThe airbnb rent prediction per night for below features is:")
pred = predict2(Property_type, Room_type, Accommodates, Bathrooms, Bed_type, Cancellation_policy, Bedrooms, Beds)

print("Property_type        :", Property_type)
print("Room_type            :", Room_type)
print("Accommodates         :", Accommodates)
print("Bathrooms            :", Bathrooms)
print("Bed_type             :", Bed_type)
print("Cancellation_policy  :", Cancellation_policy)
print("Bedrooms             :", Bedrooms)
print("Beds                 :", Beds)
