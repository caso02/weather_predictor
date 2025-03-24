import csv
import os

class WeatherPipeline:
    def __init__(self):
        self.file_path = os.path.join('data', 'weather_history.csv')
        self.file = None
        self.writer = None

    def open_spider(self, spider):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        # Open the file in write mode
        self.file = open(self.file_path, 'w', newline='', encoding='utf-8')
        # Define the CSV writer with the fields specified in settings.py
        self.writer = csv.DictWriter(self.file, fieldnames=spider.settings.get('FEED_EXPORT_FIELDS'))
        # Write the header
        self.writer.writeheader()

    def close_spider(self, spider):
        if self.file:
            self.file.close()

    def process_item(self, item, spider):
        # Write each item to the CSV file
        self.writer.writerow(item)
        return item

# Keep the existing CryptoPipeline if you still need it
class CryptoPipeline:
    def __init__(self):
        self.file_path = os.path.join('scrapy_project', 'data', 'BTC_history.csv')
        self.file = None
        self.writer = None

    def open_spider(self, spider):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self.file = open(self.file_path, 'w', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.file, fieldnames=['Date', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume'])
        self.writer.writeheader()

    def close_spider(self, spider):
        if self.file:
            self.file.close()

    def process_item(self, item, spider):
        self.writer.writerow(item)
        return item