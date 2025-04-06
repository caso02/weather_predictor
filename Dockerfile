# Verwende ein schlankeres Python-Image als Basis
FROM python:3.13-slim

# Setze das Arbeitsverzeichnis
WORKDIR /usr/src/app

# Installiere Build-Tools für numpy und entferne sie anschließend
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Kopiere zuerst die requirements.txt und installiere Abhängigkeiten
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiere den restlichen Code
COPY app.py app.py
COPY frontend/index.html frontend/index.html
COPY scrapy_project/data/weather_history.csv scrapy_project/data/weather_history.csv
COPY model/train_model.py model/train_model.py
COPY model/save.py model/save.py

# Setze Umgebungsvariablen für Flask
ENV FLASK_APP=/usr/src/app/app.py
ENV FLASK_ENV=production

# Exponiere Port 80
EXPOSE 80

# Starte die Flask-App
CMD ["flask", "run", "--host=0.0.0.0", "--port=80"]