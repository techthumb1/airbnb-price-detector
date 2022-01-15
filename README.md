
# Airbnb Price Detector

Using historical booking data from AirBnB, AirBnB Optimal Price will allow a user to predict the trends for optimal pricing for their properties based on variables of location, time of year, and other considerations.


## Technical Details
- Python
- FastAPI
- Flake8
- Heroku
- Plotly
- Pytest


## Installation


```
pipenv install dev
```

Activate the virtual environment:


```
pipenv shell
```


## Launching the application


```
uvicorn app.main:app --reload
```

Application is now running on localhost:8000.


## Testing the application

```
pipenv run pytest
```


## Usage
The following is a sample of the API call:
    
    http://localhost:8000/api/v1/price_prediction?location=San%20Francisco&check_in=2020-01-01&check_out=2020-01-02&guests=2


## API Endpoints

    - /api/v1/price_prediction
    - /api/v1/price_prediction/help
    - /api/v1/price_prediction/version


## API Documentation

    - /api/v1/price_prediction/help
    - /api/v1/price_prediction/version




## Deployment
 


## Source Code


## License


## Contributors


## Tests


## Future Plans


## Contact

 - [robinsonjason761@gmail.com]