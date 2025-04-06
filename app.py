from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os
import subprocess
from datetime import datetime
import pickle
from pymongo import MongoClient
import sys

# Initialize Flask app with the correct template folder
app = Flask(__name__, template_folder='frontend')

# MongoDB connection (use environment variable for the full URI)
MONGODB_URI = os.getenv('MONGODB_URI')
if not MONGODB_URI:
    raise ValueError("MONGODB_URI environment variable not set")

client = MongoClient(MONGODB_URI)
db = client['weather_db']
collection = db['weather_history']

# Path to the weather_history.csv file
WEATHER_CSV_PATH = os.path.join('scrapy_project', 'data', 'weather_history.csv')
# Path to the models directory
MODELS_DIR = 'model/models'
# Ensure the models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)
# Path to the prediction results
PREDICTION_PATH = os.path.join(MODELS_DIR, 'predictions.pkl')

def load_csv_to_db():
    """Read the CSV file, process the data, and load it into MongoDB."""
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
        month_map = {
            'Jan': '01', 'Feb': '02', 'Mär': '03', 'Apr': '04', 'Mai': '05', 'Jun': '06',
            'Jul': '07', 'Aug': '08', 'Sep': '09', 'Okt': '10', 'Nov': '11', 'Dez': '12'
        }
        day, month = date_str.split('. ')
        day = day.strip()
        month = month_map[month.strip()]
        date_obj = datetime.strptime(f"2025-{month}-{day.zfill(2)}", "%Y-%m-%d")
        day_map = {0: 'Mo', 1: 'Di', 2: 'Mi', 3: 'Do', 4: 'Fr', 5: 'Sa', 6: 'So'}
        return day_map[date_obj.weekday()]

    df['Day'] = df['Date'].apply(extract_day)

    # Convert DataFrame to list of dictionaries for MongoDB
    records = df.to_dict('records')

    # Check if the collection already has data
    if collection.count_documents({}) > 0:
        app.logger.info("Collection 'weather_history' already has data. Skipping CSV load.")
        return True

    # Insert data into MongoDB
    collection.insert_many(records)
    app.logger.info("Data successfully loaded into MongoDB.")
    return True

@app.route('/')
def index():
    if not load_csv_to_db():
        return "Error: Could not load data from CSV file. Please ensure weather_history.csv exists.", 500

    try:
        # Fetch unique dates
        unique_dates = collection.distinct('Date')
        app.logger.info(f"Unique dates (unsorted): {unique_dates}")

        # Function to convert date string to datetime for sorting
        def parse_date(date_str):
            month_map = {
                'Jan': '01', 'Feb': '02', 'Mär': '03', 'Apr': '04', 'Mai': '05', 'Jun': '06',
                'Jul': '07', 'Aug': '08', 'Sep': '09', 'Okt': '10', 'Nov': '11', 'Dez': '12'
            }
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
            data = list(collection.find({'Date': selected_date}).sort('Time'))
            app.logger.info(f"Filtered data for {selected_date}: {len(data)} rows")
        else:
            data = list(collection.find().sort([('Date', -1), ('Time', 1)]))
            app.logger.info("No date selected, showing all data")

        # Load prediction results if available
        prediction = None
        days_ahead = request.args.get('days_ahead', None)  # Get days_ahead from query parameter
        if os.path.exists(PREDICTION_PATH):
            try:
                with open(PREDICTION_PATH, 'rb') as f:
                    predictions = pickle.load(f)
                app.logger.info(f"Prediction loaded: {predictions}")
                # Find the prediction for the selected days_ahead
                if days_ahead:
                    days_ahead = int(days_ahead)
                    for pred in predictions:
                        if pred['days_ahead'] == days_ahead:
                            prediction = pred
                            break
            except Exception as e:
                app.logger.error(f"Error loading prediction: {str(e)}")

        return render_template('index.html', data=data, unique_dates=unique_dates, selected_date=selected_date, prediction=prediction, days_ahead=days_ahead)

    except Exception as e:
        app.logger.error(f"Error accessing MongoDB: {str(e)}")
        return f"Error accessing MongoDB: {str(e)}", 500

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

        # Redirect back to the index page with the selected date and days_ahead
        selected_date = request.args.get('date', '')
        return redirect(url_for('index', date=selected_date, days_ahead=days_ahead))

    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return f"Error during prediction: {str(e)}", 500

# Allow running load_csv_to_db() directly from the command line
if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'load_csv_to_db':
        load_csv_to_db()
    else:
        app.run(debug=True, port=5000)