import scrapy
from scrapy_project.items import WeatherItem
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
import time
from datetime import datetime

class WeatherSpider(scrapy.Spider):
    name = "weather_spider"
    allowed_domains = ['timeanddate.de']
    start_urls = ['https://www.timeanddate.de/wetter/schweiz/zuerich/rueckblick']

    def __init__(self, *args, **kwargs):
        super(WeatherSpider, self).__init__(*args, **kwargs)
        # Set up Selenium WebDriver
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        self.driver = webdriver.Chrome(options=chrome_options)

    def parse(self, response):
        # Use Selenium to load the page
        self.driver.get(response.url)

        try:
            # Wait for the dropdown to be present
            dropdown = WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.ID, 'wt-his-select'))
            )
            # Create a Select object to interact with the dropdown
            select = Select(dropdown)

            # Get all available date options (excluding "Letzte 24 Stunden")
            date_options = []
            for option in select.options:
                value = option.get_attribute('value')
                text = option.text
                self.logger.debug(f"Option - Value: {value}, Text: {text}")
                if value and value != '' and text != 'Letzte 24 Stunden':
                    date_options.append(value)

            self.logger.info(f"Filtered date options: {date_options}")

            # Iterate through each date
            for date_value in date_options:
                try:
                    # Select the date from the dropdown
                    self.logger.info(f"Selecting date: {date_value}")
                    select.select_by_value(date_value)
                    self.logger.info(f"Selected date: {date_value}")

                    # Convert the date_value (e.g., "20250320") to display format (e.g., "20. Mär")
                    date_obj = datetime.strptime(date_value, "%Y%m%d")
                    month_map = {1: 'Jan', 2: 'Feb', 3: 'Mär', 4: 'Apr', 5: 'Mai', 6: 'Jun',
                                 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Okt', 11: 'Nov', 12: 'Dez'}
                    display_date = f"{date_obj.day}. {month_map[date_obj.month]}"
                    self.logger.info(f"Display date: {display_date}")

                    # Wait for the table to update
                    WebDriverWait(self.driver, 20).until(
                        EC.presence_of_element_located((By.XPATH, '//table[@id="wt-his"]//tbody/tr'))
                    )

                    # Parse the table rows
                    rows = self.driver.find_elements(By.XPATH, '//table[@id="wt-his"]//tbody/tr')
                    self.logger.info(f"Found {len(rows)} rows for date {date_value}")
                    for row in rows:
                        try:
                            cells = row.find_elements(By.TAG_NAME, 'td')
                            time_cell = row.find_element(By.TAG_NAME, 'th')

                            # Extract the time (e.g., "17:00")
                            time_text = time_cell.text
                            time = time_text.split('\n')[0]  # Extract time part (e.g., "17:00")

                            item = WeatherItem()
                            item['Date'] = display_date  # Use the display date for all rows
                            item['Time'] = time
                            item['Temperature'] = cells[1].text  # Temp
                            item['Weather'] = cells[2].text  # Weather
                            item['Wind_Speed'] = cells[3].text  # Wind speed
                            item['Wind_Direction'] = cells[4].find_element(By.TAG_NAME, 'span').get_attribute('title')  # Wind direction
                            item['Humidity'] = cells[5].text  # Humidity
                            item['Barometer'] = cells[6].text  # Barometer
                            item['Visibility'] = cells[7].text  # Visibility
                            yield item
                        except StaleElementReferenceException as e:
                            self.logger.warning(f"Stale element reference while parsing row for date {date_value}: {str(e)}")
                            continue

                    # Add a delay to avoid potential anti-bot measures
                    time.sleep(2)

                except TimeoutException as e:
                    self.logger.error(f"Timeout while processing date {date_value}: {str(e)}")
                    continue
                except Exception as e:
                    self.logger.error(f"Error while processing date {date_value}: {str(e)}")
                    continue

        except TimeoutException as e:
            self.logger.error(f"Timeout while loading page or dropdown: {str(e)}")
        finally:
            self.driver.quit()