from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import sqlite3
import os
import subprocess
from datetime import datetime
import pickle

# Initialize Flask app with the correct template folder
app = Flask(__name__, template_folder='frontend')

# Path to the weather_history.csv file
WEATHER_CSV_PATH = os.path.join('scrapy_project', 'data', 'weather_history.csv')
# Path to the SQLite database
DATABASE_PATH = 'weather.db'  # Relative path in the root directory
# Path to the models directory
MODELS_DIR = 'model/models'  # Relative path to model/models/
# Ensure the models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)
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
        app.logger.error(f"weather_history.csv not found at {os.path.abspath(WEATHER_CSV_PATH)}. Please run the Scrapy spider to generate the file.")
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

    # Only delete the table contents if we're sure we can load new data
    cursor.execute("SELECT COUNT(*) FROM weather_history")
    count = cursor.fetchone()[0]
    if count > 0:
        app.logger.info("Table 'weather_history' already has data. Skipping CSV load.")
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
    app.logger.info("Data successfully loaded into SQLite database.")
    return True

@app.route('/')
def index():
    init_db()
    if not load_csv_to_db():
        return "Error: Could not load data from CSV file. Please ensure weather_history.csv exists.", 500

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

    except sqlite3.Error as e:
        app.logger.error(f"SQLite database error: {str(e)}")
        return f"SQLite database error: {str(e)}", 500
    except Exception as e:
        app.logger.error(f"Error rendering template: {str(e)}")
        return f"Error rendering template: {str(e)}", 500

@app.route('/predict', methods=['POST'])
def predict():
    """Trigger the prediction for the specified day by running train_model.py."""
    try:
        # Get the number of days ahead from the form
        days_ahead = request.form.get('days_ahead', '1')  # Default to 1 if not provided
        app.logger.info(f"Predicting weather for {days_ahead} days ahead")

        # Run the train_model.py script with the days_ahead argument
        result = subprocess.run(['python', 'model/train_model.py', '--days-ahead', days_ahead], capture_output=True, text=True)
        app.logger.info(f"train_model.py output: {result.stdout}")
        app.logger.info(f"train_model.py errors: {result.stderr}")

        if result.returncode != 0:
            app.logger.error(f"Error running train_model.py: {result.stderr}")
            return f"Error running prediction script: {result.stderr}", 500

        # Load the prediction result
        if os.path.exists(PREDICTION_PATH):
            try:
                with open(PREDICTION_PATH, 'rb') as f:
                    prediction = pickle.load(f)
                app.logger.info(f"Prediction loaded: {prediction}")
            except Exception as e:
                app.logger.error(f"Error loading prediction: {str(e)}")
                return f"Error loading prediction: {str(e)}", 500
        else:
            app.logger.error("Prediction file not found after running train_model.py.")
            return "Prediction file not found after running train_model.py.", 500

        # Redirect back to the index page with the selected date (if any)
        selected_date = request.args.get('date', '')
        return redirect(url_for('index', date=selected_date))

    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return f"Error during prediction: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)