import pandas as pd
from nixtlats import TimeGPT
import matplotlib.pyplot as plt

# read .env token
import os
from dotenv import load_dotenv

# Assuming your .env file is located in the same directory as your Python script
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")

# Load the variables from the .env file
load_dotenv(dotenv_path)

# Access the token from the environment variables
token = os.getenv("TOKEN")

# Initializing the `TimeGPT` class with a token from the environment variable.
timegpt = TimeGPT(token=token)  # https://dashboard.nixtla.io

# Check if the token provided is valid.
if timegpt.validate_token():
    print("Token validation successful!")  # Token is valid.
else:
    # Raise an exception if token validation fails.
    raise Exception(
        "Token validation failed! Please check go to https://dashboard.nixtla.io/ to get your token."
    )

# Loading the air passengers dataset from a remote URL as an example
df = pd.read_csv(
    "https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/main/datasets/air_passengers.csv"
)

# Forecasting the next 12 horizons using TimeGPT
timegpt_fcst_df = timegpt.forecast(
    df=df, h=12, time_col="timestamp", target_col="value"
)

# Plotting the original data combined with the forecasted data.
pd.concat([df, timegpt_fcst_df]).set_index("timestamp").plot()
