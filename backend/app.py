from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import sqlite3
import os
from datetime import datetime
import pickle
from model.train_model import run_prediction  # Import the prediction function

app = Flask(__name__)

# Path to the weather_history.csv file
WEATHER_CSV_PATH = os.path.join('scrapy_project', 'data', 'weather_history.csv')
# Path to the SQLite database
DATABASE_PATH = 'weather.db'
# Path to the models directory
MODELS_DIR = 'models'
# Path to the prediction results
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
        app.logger.error("weather_history.csv not found.")
        return False

    try:
        df = pd.read_csv(WEATHER_CSV_PATH)
        app.logger.info(f"CSV file read successfully. Number of rows: {len(df)}")
    except Exception as e:
        app.logger.error(f"Error reading weather_history.csv: {str(e)}")
        return False

    # Process the Date column to add the Day column
    def extract_day(date_str):
        month_map = {'Mär': '03'}
        day, month = date_str.split('. ')
        day = day.strip()
        month = month_map[month.strip()]
        date_obj = datetime.strptime(f"2025-{month}-{day.zfill(2)}", "%Y-%m-%d")
        day_map = {0: 'Mo', 1: 'Di', 2: 'Mi', 3: 'Do', 4: 'Fr', 5: 'Sa', 6: 'So'}
        return day_map[date_obj.weekday()]

    df['Day'] = df['Date'].apply(extract_day)

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

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
    app.logger.info("Data successfully loaded into SQLite database.")
    return True

@app.route('/')
def index():
    init_db()
    if not load_csv_to_db():
        return "Error: Could not load data from CSV file.", 500

    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()

        # Fetch unique dates
        cursor.execute("SELECT DISTINCT Date FROM weather_history")
        unique_dates = [row[0] for row in cursor.fetchall()]
        app.logger.info(f"Unique dates (unsorted): {unique_dates}")

        # Function to convert date string to datetime for sorting
        def parse_date(date_str):
            month_map = {'Mär': '03'}
            day, month = date_str.split('. ')
            day = day.strip()
            month = month_map[month.strip()]
            return datetime.strptime(f"2025-{month}-{day.zfill(2)}", "%Y-%m-%d")

        # Sort dates in descending chronological order
        unique_dates.sort(key=parse_date, reverse=True)
        app.logger.info(f"Unique dates (sorted descending): {unique_dates}")

        # Get the selected date from the query parameter (default to the most recent date)
        selected_date = request.args.get('date', unique_dates[0] if unique_dates else None)
        app.logger.info(f"Selected date from query parameter: {selected_date}")

        # Fetch the data for the selected date
        if selected_date:
            cursor.execute("SELECT * FROM weather_history WHERE Date = ? ORDER BY Time", (selected_date,))
            data = cursor.fetchall()
            app.logger.info(f"Filtered data for {selected_date}: {len(data)} rows")
        else:
            cursor.execute("SELECT * FROM weather_history ORDER BY Date DESC, Time")
            data = cursor.fetchall()
            app.logger.info("No date selected, showing all data")

        columns = ['Date', 'Time', 'Temperature', 'Weather', 'Wind_Speed', 'Wind_Direction', 'Humidity', 'Barometer', 'Visibility', 'Day']
        data = [dict(zip(columns, row)) for row in data]

        # Load prediction results if available
        prediction = None
        if os.path.exists(PREDICTION_PATH):
            try:
                with open(PREDICTION_PATH, 'rb') as f:
                    prediction = pickle.load(f)
                app.logger.info(f"Prediction loaded: {prediction}")
            except Exception as e:
                app.logger.error(f"Error loading prediction: {str(e)}")

        conn.close()

        return render_template('index.html', data=data, unique_dates=unique_dates, selected_date=selected_date, prediction=prediction)

    except Exception as e:
        app.logger.error(f"Error accessing SQLite database: {str(e)}")
        return f"Error accessing SQLite database: {str(e)}", 500

@app.route('/predict', methods=['POST'])
def predict():
    """Trigger the prediction for the next day by calling the prediction function."""
    try:
        # Run the prediction
        predictions = run_prediction()
        app.logger.info(f"Prediction completed: {predictions}")

        # Redirect back to the index page with the selected date (if any)
        selected_date = request.args.get('date', '')
        return redirect(url_for('index', date=selected_date))

    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return f"Error during prediction: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)