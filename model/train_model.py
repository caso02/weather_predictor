import pandas as pd
import sqlite3
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import pickle
import os

# Path to the SQLite database
DATABASE_PATH = 'weather.db'
# Path to the models directory
MODELS_DIR = 'models'
# Ensure the models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)
# Paths for saving/loading files
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, 'weather_label_encoder.pkl')
TEMP_MODEL_PATH = os.path.join(MODELS_DIR, 'temp_model.pkl')
WEATHER_MODEL_PATH = os.path.join(MODELS_DIR, 'weather_model.pkl')
PREDICTION_PATH = os.path.join(MODELS_DIR, 'predictions.pkl')

def load_data():
    """Load data from the SQLite database."""
    conn = sqlite3.connect(DATABASE_PATH)
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

    # Encode Weather (for classification)
    label_encoder = LabelEncoder()
    df['Weather_Encoded'] = label_encoder.fit_transform(df['Weather'])

    # Save the label encoder
    with open(LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)

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

def train_models(df):
    """Train Random Forest models for temperature and weather condition prediction."""
    # Features for both models
    features = ['Wind_Speed', 'Wind_Direction_Degrees', 'Humidity', 'Barometer', 'Visibility', 'Hour', 'Day_of_Week', 'Day_of_Month']

    # Temperature Prediction (Regression)
    X_temp = df[features]
    y_temp = df['Temperature']
    X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X_temp, y_temp, test_size=0.2, shuffle=False)

    temp_model = RandomForestRegressor(n_estimators=100, random_state=42)
    temp_model.fit(X_train_temp, y_train_temp)
    y_pred_temp = temp_model.predict(X_test_temp)
    rmse = np.sqrt(mean_squared_error(y_test_temp, y_pred_temp))
    print(f"Temperature Prediction RMSE: {rmse}")

    # Weather Condition Prediction (Classification)
    X_weather = df[features]
    y_weather = df['Weather_Encoded']
    X_train_weather, X_test_weather, y_train_weather, y_test_weather = train_test_split(X_weather, y_weather, test_size=0.2, shuffle=False)

    weather_model = RandomForestClassifier(n_estimators=100, random_state=42)
    weather_model.fit(X_train_weather, y_train_weather)
    y_pred_weather = weather_model.predict(X_test_weather)
    accuracy = accuracy_score(y_test_weather, y_pred_weather)
    print(f"Weather Condition Prediction Accuracy: {accuracy}")
    print(classification_report(y_test_weather, y_pred_weather))

    # Save the models
    with open(TEMP_MODEL_PATH, 'wb') as f:
        pickle.dump(temp_model, f)
    with open(WEATHER_MODEL_PATH, 'wb') as f:
        pickle.dump(weather_model, f)

    return temp_model, weather_model

def predict_next_day(df, temp_model, weather_model):
    """Predict the weather for the next day (March 21, 2025)."""
    # Features for prediction
    features = ['Wind_Speed', 'Wind_Direction_Degrees', 'Humidity', 'Barometer', 'Visibility', 'Hour', 'Day_of_Week', 'Day_of_Month']

    # Get the most recent day's data (March 20, 2025)
    last_day = df[df['Date'] == '20. Mär'].copy()

    # Prepare data for March 21, 2025
    prediction_data = last_day[features].copy()
    # Update Day_of_Month to 21
    prediction_data['Day_of_Month'] = 21
    # March 21, 2025, is a Friday (Day_of_Week = 4)
    prediction_data['Day_of_Week'] = 4

    # Predict temperature (average over all hours)
    temp_pred = temp_model.predict(prediction_data[features])
    avg_temp = np.mean(temp_pred)
    print(f"Predicted average temperature for March 21, 2025: {avg_temp:.1f} °C")

    # Predict weather condition (most common prediction over all hours)
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    weather_pred = weather_model.predict(prediction_data[features])
    most_common_weather = label_encoder.inverse_transform([np.bincount(weather_pred).argmax()])[0]
    print(f"Predicted weather condition for March 21, 2025: {most_common_weather}")

    # Save the predictions
    predictions = {
        'avg_temperature': avg_temp,
        'weather_condition': most_common_weather
    }
    with open(PREDICTION_PATH, 'wb') as f:
        pickle.dump(predictions, f)

    return predictions

def train_and_save_models():
    """Load data, preprocess it, train models, and save them."""
    df = load_data()
    df = preprocess_data(df)
    temp_model, weather_model = train_models(df)
    return temp_model, weather_model

def run_prediction():
    """Run the prediction for the next day using saved models."""
    # Load the data
    df = load_data()
    df = preprocess_data(df)

    # Load the trained models
    with open(TEMP_MODEL_PATH, 'rb') as f:
        temp_model = pickle.load(f)
    with open(WEATHER_MODEL_PATH, 'rb') as f:
        weather_model = pickle.load(f)

    # Predict the weather for the next day
    predictions = predict_next_day(df, temp_model, weather_model)
    return predictions

if __name__ == "__main__":
    # Train and save the models when the script is run directly
    temp_model, weather_model = train_and_save_models()
    # Do not run the prediction automatically
    print("Models trained and saved. Run the prediction via the Flask app.")