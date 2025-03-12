# Demand-Sale-Forecasting-ML-Model using Prophet Model


## Project Overview
•	Objective: Predict future sales demand based on past sales, seasonality, and promotions.
•	Tools: Python (pandas, statsmodels, scikit-learn, Prophet), SQL for data extraction, Flask/FastAPI for deployment, and cloud options like AWS/GCP/Azure for production.


# Project Roadmap
1. Data Collection & Preprocessing
•	Load historical sales data from SQL or CSV.
•	Handle missing values, outliers, and feature engineering (seasonality, holidays, promotions).
•	Train-test split.

2. Exploratory Data Analysis (EDA)
•	Analyze trends, seasonality, and promotional impact.
•	Visualize using seaborn, matplotlib, and Plotly.

3. Model Building
•	Machine Learning: Random Forest, XGBoost for forecasting.
•	Deep Learning (Optional): LSTMs if needed.
•	Prophet Model: Facebook's Prophet for easy handling of holidays and seasonality.

4. Model Evaluation
•	Use RMSE, MAE, MAPE, and R² for performance evaluation.

5. Deployment
•	Build an API using Flask/FastAPI.



# CODE

import pandas as pd

import numpy as np

df=pd.read_csv("demand_prediction_dataset.csv")


# check dataset info
df.info()

result see

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 300 entries, 0 to 299
Data columns (total 7 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   product_id    300 non-null    object 
 1   product_name  300 non-null    object 
 2   date          300 non-null    object 
 3   price (INR)   300 non-null    float64
 4   promotion     300 non-null    int64  
 5   holiday       300 non-null    int64  
 6   seasonality   300 non-null    object 
dtypes: float64(1), int64(2), object(4)
memory usage: 16.5+ KB



# cleaning and transformation apply

#your date format is "day-month-year" (1-10-2024), and pd.to_datetime() 
#might not be recognizing it correctly. By default, pandas expects "year-month-day" (YYYY-MM-DD) format.so you have define format here

df["date"]=pd.to_datetime(df["date"],format="%d-%m-%Y")


# set date as index for easy accesing the value 

df.set_index(df["date"])

df.sort_values('date')

df.rename(columns={"price (INR)":"sale"},inplace=True)



# for prophet model , there is no need of one hot encoding and min-max scaler ,becuase it not effect by feature magnitudue
!pip install prophet

# preparing the data for prophet model

prophet_df=df[["date","sale"]]

#prophet data take datetime as ds input, which we want to predict take it  as y

prophet_df.rename(columns={"date":"ds","sale":"y"},inplace=True)


#applying the model
from prophet import Prophet

import matplotlib.pyplot as plt

#intialisation of model 

model=Prophet(seasonality_mode='multiplicative')

#train the model

model.fit(prophet_df)

                  



<prophet.forecaster.Prophet at 0x1494d722950>




# Create future dataframe for predictions--model need date of next 30 day for prediction , so we create it
future = model.make_future_dataframe(periods=30)

forecast = model.predict(future)

 #Plot the forecast
 
model.plot(forecast)

plt.title('Prophet Model Forecast with Seasonality')

plt.show()#


# save this model 
import pickle

with open ("sale_prediction_model.pkl","wb") as f:

    pickle.dump(model,f)

    


    # deploying an model , you can delploy model using FASTAPI ,FLASKAPI , web front end application integration with model and on cloud server

    app = Flask(__name__)

# Load the saved Prophet model
with open("sale_prediction_model.pkl", "rb") as f:

    model = pickle.load(f)
    
@app.route('/',methods=['GET'])

def home():

    return "Hello, Flask with Prophet!"
    
@app.route('/predict', methods=['POST'])

def predict():

    data = request.get_json()  # Expect JSON input with future dates
    
    future_dates = pd.DataFrame(data["dates"], columns=["ds"])
    
    forecast = model.predict(future_dates)
    
    return jsonify(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_dict(orient="records"))

if __name__ == "__main__":

    app.run(debug=True)





  


#your model is host on Flask API server , you must know how to start Flask API server using CMD 

  1. Navigate to Your Project Folder
       cd your project folder path , where flask app.py is available, run this command
 2..Activate Virtual Environment
       venv\Scripts\activate      run this command
 3..Run the Flask App
        python app.py




# after starting ,it cmd will give this result ,it working fine 
Microsoft Windows [Version 10.0.22631.3737]
(c) Microsoft Corporation. All rights reserved.

C:\Users\amitn>cd C:\Users\amitn\Documents\flask_project

C:\Users\amitn\Documents\flask_project>venv\Scripts\activate

(venv) C:\Users\amitn\Documents\flask_project>python app.py
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 923-999-212



   




    






•	Deploy on AWS Lambda/Azure Functions/GCP Cloud Run.
•	Integrate with a web app or Power BI dashboard.

