# Use the official Python 3.13 image as the base
FROM python:3.13.0

# Set the working directory
WORKDIR /usr/src/app

# Copy the necessary files
COPY backend/app.py backend/app.py
COPY frontend/index.html frontend/index.html
COPY scrapy_project/data/weather_history.csv scrapy_project/data/weather_history.csv
COPY model/train_model.py model/train_model.py
COPY model/save.py model/save.py

# Copy the requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Expose port 80
EXPOSE 80

# Set the Flask app environment variable
ENV FLASK_APP=/usr/src/app/backend/app.py

# Command to run the Flask app
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=80"]