BOT_NAME = 'scrapy_project'

SPIDER_MODULES = ['scrapy_project.spiders']
NEWSPIDER_MODULE = 'scrapy_project.spiders'

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'

ROBOTSTXT_OBEY = False

DOWNLOAD_DELAY = 3
CONCURRENT_REQUESTS = 1

FEED_EXPORT_ENCODING = 'utf-8'

COOKIES_ENABLED = True

DOWNLOADER_MIDDLEWARES = {
    'scrapy.downloadermiddlewares.httpproxy.HttpProxyMiddleware': 110,
    # Removed 'scrapy_selenium.SeleniumMiddleware' since we're managing Selenium directly in the spider
}

ITEM_PIPELINES = {
    'scrapy_project.pipelines.WeatherPipeline': 300,
}

# Ensure CSV output uses commas as the delimiter
FEED_EXPORT_FIELDS = ['Date', 'Time', 'Temperature', 'Weather', 'Wind_Speed', 'Wind_Direction', 'Humidity', 'Barometer', 'Visibility']
FEED_EXPORT_INDENT = 0
FEED_EXPORT_PARAMETERS = {
    'delimiter': ',',  # Ensure comma is used as the delimiter
}