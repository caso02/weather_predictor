import scrapy

class WeatherItem(scrapy.Item):
    Date = scrapy.Field()
    Time = scrapy.Field()
    Temperature = scrapy.Field()
    Weather = scrapy.Field()
    Wind_Speed = scrapy.Field()
    Wind_Direction = scrapy.Field()
    Humidity = scrapy.Field()
    Barometer = scrapy.Field()
    Visibility = scrapy.Field()