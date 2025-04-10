import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle
import os
import argparse
from datetime import datetime, timedelta
from pymongo import MongoClient

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train and predict weather for a specified day.')
parser.add_argument('--days-ahead', type=int, default=1, help='Number of days ahead to predict (default: 1)')
args = parser.parse_args()

# MongoDB connection (use environment variable for security in production)
MONGODB_URI = os.getenv('MONGODB_URI')
if not MONGODB_URI:
    raise ValueError("MONGODB_URI environment variable not set")

client = MongoClient(MONGODB_URI)
db = client['weather_db']
collection = db['weather_history']

# Path to the weather_history.csv file
WEATHER_CSV_PATH = os.path.join('scrapy_project', 'data', 'weather_history.csv')
# Path to the models directory
MODELS_DIR = 'model/models'  # Relative to the root directory
# Ensure the models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)
# Paths for saving/loading files
TEMP_MODEL_PATH = os.path.join(MODELS_DIR, 'temp_model.pkl')
PREDICTION_PATH = os.path.join(MODELS_DIR, 'predictions.pkl')

def load_csv_to_db():
    """Read the CSV file, process the data, and load it into MongoDB."""
    if not os.path.exists(WEATHER_CSV_PATH):
        print(f"Error: {os.path.abspath(WEATHER_CSV_PATH)} not found. Please run the Scrapy spider to generate the file.")
        return False

    try:
        df = pd.read_csv(WEATHER_CSV_PATH)
        print(f"CSV file read successfully. Number of rows: {len(df)}")
    except Exception as e:
        print(f"Error reading weather_history.csv: {str(e)}")
        return False

    # Process the Date column to add the Day column
    def extract_day(date_str):
        month_map = {
            'Jan': '01', 'Feb': '02', 'Mär': '03', 'Apr': '04', 'Mai': '05', 'Jun': '06',
            'Jul': '07', 'Aug': '08', 'Sep': '09', 'Okt': '10', 'Nov': '11', 'Dez': '12'
        }
        day, month = date_str.split('. ')
        day = day.strip()
        month = month_map[month.strip()]
        date_obj = pd.to_datetime(f"2025-{month}-{day.zfill(2)}", format="%Y-%m-%d")
        day_map = {0: 'Mo', 1: 'Di', 2: 'Mi', 3: 'Do', 4: 'Fr', 5: 'Sa', 6: 'So'}
        return day_map[date_obj.dayofweek]

    df['Day'] = df['Date'].apply(extract_day)

    # Convert DataFrame to list of dictionaries for MongoDB
    records = df.to_dict('records')

    # Check if the collection already has data
    if collection.count_documents({}) > 0:
        print("Collection 'weather_history' already has data. Skipping CSV load.")
        return True

    # Insert data into MongoDB
    collection.insert_many(records)
    print("Data successfully loaded into MongoDB.")
    return True

def load_data():
    """Load data from MongoDB."""
    # Load the CSV data if the collection doesn't exist or is empty
    if collection.count_documents({}) == 0:
        print("Collection 'weather_history' is empty. Loading data from CSV...")
        if not load_csv_to_db():
            raise Exception("Failed to load data from CSV into the database: CSV file not found or invalid.")

    # Fetch all data from MongoDB and convert to DataFrame
    data = list(collection.find().sort([('Date', 1), ('Time', 1)]))
    df = pd.DataFrame(data)
    return df

def preprocess_data(df):
    """Preprocess the data for training."""
    # Clean Temperature
    df['Temperature'] = df['Temperature'].str.replace(' °C', '').astype(float)

    # Clean Wind Speed
    df['Wind_Speed'] = df['Wind_Speed'].replace('Windstill', '0 km/h')
    df['Wind_Speed'] = df['Wind_Speed'].str.replace(' km/h', '').astype(float)

    # Extract Wind Direction (degrees)
    df['Wind_Direction_Degrees'] = df['Wind_Direction'].str.extract(r'(\d+)°').astype(float)

    # Clean Humidity
    df['Humidity'] = df['Humidity'].str.replace('%', '').astype(float)

    # Clean Barometer
    df['Barometer'] = df['Barometer'].str.replace(' hPa', '').astype(float)

    # Clean Visibility
    df['Visibility'] = df['Visibility'].replace('k.A.', np.nan)
    df['Visibility'] = df['Visibility'].str.replace(' km', '').astype(float)
    df['Visibility'] = df['Visibility'].fillna(df['Visibility'].mean())

    # Extract Hour from Time
    df['Hour'] = df['Time'].str.split(':').str[0].astype(int)

    # Convert Date to datetime and extract features
    def parse_date(date_str):
        month_map = {
            'Jan': '01', 'Feb': '02', 'Mär': '03', 'Apr': '04', 'Mai': '05', 'Jun': '06',
            'Jul': '07', 'Aug': '08', 'Sep': '09', 'Okt': '10', 'Nov': '11', 'Dez': '12'
        }
        try:
            day, month = date_str.split('. ')
            day = day.strip()
            month = month_map[month.strip()]
            return pd.to_datetime(f"2025-{month}-{day.zfill(2)}", format="%Y-%m-%d")
        except KeyError as e:
            raise ValueError(f"Invalid month in date string '{date_str}': {str(e)}")
        except Exception as e:
            raise ValueError(f"Error parsing date '{date_str}': {str(e)}")

    df['Datetime'] = df['Date'].apply(parse_date)
    df['Day_of_Month'] = df['Datetime'].dt.day
    df['Day_of_Week'] = df['Datetime'].dt.dayofweek

    return df

def train_model(df):
    """Train a Random Forest model for temperature prediction."""
    # Features for the model
    features = ['Wind_Speed', 'Wind_Direction_Degrees', 'Humidity', 'Barometer', 'Visibility', 'Hour', 'Day_of_Week', 'Day_of_Month']

    # Temperature Prediction (Regression)
    X = df[features]
    y = df['Temperature']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Temperature Prediction RMSE: {rmse}")

    # Save the model
    with open(TEMP_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    return model

def predict_next_day(df, model, days_ahead):
    """Predict the temperature for the specified day ahead."""
    # Features for prediction
    features = ['Wind_Speed', 'Wind_Direction_Degrees', 'Humidity', 'Barometer', 'Visibility', 'Hour', 'Day_of_Week', 'Day_of_Month']

    # Debug: Print all dates in the dataset
    print("All dates in dataset:", sorted(df['Date'].unique()))

    # Dynamically determine the most recent date in the dataset
    def parse_date(date_str):
        month_map = {
            'Jan': '01', 'Feb': '02', 'Mär': '03', 'Apr': '04', 'Mai': '05', 'Jun': '06',
            'Jul': '07', 'Aug': '08', 'Sep': '09', 'Okt': '10', 'Nov': '11', 'Dez': '12'
        }
        try:
            day, month = date_str.split('. ')
            day = day.strip()
            month = month_map[month.strip()]
            return datetime.strptime(f"2025-{month}-{day.zfill(2)}", "%Y-%m-%d")
        except KeyError as e:
            raise ValueError(f"Invalid month in date string '{date_str}': {str(e)}")
        except Exception as e:
            raise ValueError(f"Error parsing date '{date_str}': {str(e)}")

    # Convert all dates to datetime objects and find the most recent one
    df['parsed_date'] = df['Date'].apply(parse_date)
    most_recent_date = df['parsed_date'].max()
    most_recent_day = most_recent_date.day  # Get the day without leading zero
    most_recent_month = most_recent_date.strftime("%b").replace("Mär", "Mär")  # Keep month format consistent
    most_recent_date_str = f"{most_recent_day}. {most_recent_month}"  # Format: "6. Apr"

    print(f"Most recent date: {most_recent_date_str}")

    # Get the most recent day's data
    last_day = df[df['Date'] == most_recent_date_str].copy()

    # Debug: Print the filtered data
    print(f"Data for most recent date ({most_recent_date_str}):")
    print(last_day)

    # Check if last_day is empty
    if last_day.empty:
        raise ValueError(f"No data found for the most recent date '{most_recent_date_str}' in the dataset.")

    # Calculate the target date (most recent date + days_ahead)
    base_date = most_recent_date
    target_date = base_date + timedelta(days=days_ahead)
    target_date_str = target_date.strftime("%B %d, %Y")  # e.g., "April 07, 2025"

    # Prepare data for the target date
    prediction_data = last_day[features].copy()
    # Update Day_of_Month to the target day
    prediction_data['Day_of_Month'] = target_date.day
    # Update Day_of_Week (0 = Monday, 6 = Sunday)
    prediction_data['Day_of_Week'] = target_date.weekday()

    # Predict temperature (average over all hours)
    temp_pred = model.predict(prediction_data[features])
    avg_temp = np.mean(temp_pred)
    print(f"Predicted average temperature for {target_date_str}: {avg_temp:.1f} °C")

    # Load existing predictions if they exist, otherwise start with an empty list
    predictions = []
    if os.path.exists(PREDICTION_PATH):
        try:
            with open(PREDICTION_PATH, 'rb') as f:
                predictions = pickle.load(f)
        except Exception as e:
            print(f"Error loading existing predictions: {str(e)}")

    # Ensure predictions is a list
    if not isinstance(predictions, list):
        predictions = []

    # Add the new prediction to the list
    prediction = {
        'avg_temperature': avg_temp,
        'date': target_date_str,
        'days_ahead': days_ahead
    }
    predictions.append(prediction)

    # Save the updated list of predictions
    with open(PREDICTION_PATH, 'wb') as f:
        pickle.dump(predictions, f)

    return prediction

if __name__ == "__main__":
    # Load data, preprocess it, train the model, and predict when the script is run
    df = load_data()
    df = preprocess_data(df)
    model = train_model(df)
    prediction = predict_next_day(df, model, args.days_ahead)
    print(f"Prediction completed: {prediction}")