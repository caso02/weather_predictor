import pandas as pd
import sqlite3
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle
import os
import argparse
from datetime import datetime, timedelta

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train and predict weather for a specified day.')
parser.add_argument('--days-ahead', type=int, default=1, help='Number of days ahead to predict (default: 1)')
args = parser.parse_args()

# Path to the SQLite database
DATABASE_PATH = 'weather.db'  # Relative to the root directory (wetter_predictor/)
# Path to the weather_history.csv file
WEATHER_CSV_PATH = os.path.join('scrapy_project', 'data', 'weather_history.csv')
# Path to the models directory
MODELS_DIR = 'model/models'  # Relative to the root directory
# Ensure the models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)
# Paths for saving/loading files
TEMP_MODEL_PATH = os.path.join(MODELS_DIR, 'temp_model.pkl')
PREDICTION_PATH = os.path.join(MODELS_DIR, 'predictions.pkl')

def init_db():
    """Initialize the SQLite database and create the weather_history table."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS weather_history (
            Date TEXT,
            Time TEXT,
            Temperature TEXT,
            Weather TEXT,
            Wind_Speed TEXT,
            Wind_Direction TEXT,
            Humidity TEXT,
            Barometer TEXT,
            Visibility TEXT,
            Day TEXT
        )
    ''')

    conn.commit()
    conn.close()

def load_csv_to_db():
    """Read the CSV file, process the data, and load it into the SQLite database."""
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
        month_map = {'Mär': '03'}
        day, month = date_str.split('. ')
        day = day.strip()
        month = month_map[month.strip()]
        date_obj = pd.to_datetime(f"2025-{month}-{day.zfill(2)}", format="%Y-%m-%d")
        day_map = {0: 'Mo', 1: 'Di', 2: 'Mi', 3: 'Do', 4: 'Fr', 5: 'Sa', 6: 'So'}
        return day_map[date_obj.dayofweek]

    df['Day'] = df['Date'].apply(extract_day)

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # Only delete the table contents if we're sure we can load new data
    cursor.execute("SELECT COUNT(*) FROM weather_history")
    count = cursor.fetchone()[0]
    if count > 0:
        print("Table 'weather_history' already has data. Skipping CSV load.")
        conn.close()
        return True

    cursor.execute("DELETE FROM weather_history")

    for _, row in df.iterrows():
        cursor.execute('''
            INSERT INTO weather_history (Date, Time, Temperature, Weather, Wind_Speed, Wind_Direction, Humidity, Barometer, Visibility, Day)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            row['Date'], row['Time'], row['Temperature'], row['Weather'], row['Wind_Speed'],
            row['Wind_Direction'], row['Humidity'], row['Barometer'], row['Visibility'], row['Day']
        ))

    conn.commit()
    conn.close()
    print("Data successfully loaded into SQLite database.")
    return True

def load_data():
    """Load data from the SQLite database."""
    # Initialize the database and load the CSV data if the table doesn't exist
    init_db()
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # Check if the table exists and has data
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='weather_history'")
    if not cursor.fetchone():
        print("Table 'weather_history' does not exist. Loading data from CSV...")
        if not load_csv_to_db():
            raise Exception("Failed to load data from CSV into the database: CSV file not found or invalid.")

    # Check if the table has data
    cursor.execute("SELECT COUNT(*) FROM weather_history")
    count = cursor.fetchone()[0]
    if count == 0:
        print("Table 'weather_history' is empty. Loading data from CSV...")
        if not load_csv_to_db():
            raise Exception("Failed to load data from CSV into the database: CSV file not found or invalid.")

    query = "SELECT * FROM weather_history ORDER BY Date, Time"
    df = pd.read_sql_query(query, conn)
    conn.close()
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
        month_map = {'Mär': '03'}
        day, month = date_str.split('. ')
        day = day.strip()
        month = month_map[month.strip()]
        return pd.to_datetime(f"2025-{month}-{day.zfill(2)}", format="%Y-%m-%d")

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

    # Get the most recent day's data (March 20, 2025)
    last_day = df[df['Date'] == '20. Mär'].copy()

    # Calculate the target date (March 20 + days_ahead)
    base_date = datetime(2025, 3, 20)
    target_date = base_date + timedelta(days=days_ahead)
    target_date_str = target_date.strftime("%B %d, %Y")  # e.g., "March 21, 2025"

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

    # Save the prediction
    prediction = {
        'avg_temperature': avg_temp,
        'date': target_date_str
    }
    with open(PREDICTION_PATH, 'wb') as f:
        pickle.dump(prediction, f)

    return prediction

if __name__ == "__main__":
    # Load data, preprocess it, train the model, and predict when the script is run
    df = load_data()
    df = preprocess_data(df)
    model = train_model(df)
    prediction = predict_next_day(df, model, args.days_ahead)
    print(f"Prediction completed: {prediction}")