import scrapy
from urllib.parse import urlparse, parse_qs
import pandas as pd
import numpy as np


class BooksCategoriesSpider(scrapy.Spider):

    ISBN = pd.read_csv('ISBN.csv')
    ISBN_list = np.array(ISBN['ISBN'])

    name = 'books_categories'
    start_urls = [
        'http://www.bookcrossing.com/isbn?isbn={}'.format(i) for i in ISBN_list
    ]

    def parse(self, response):
        url = urlparse(response.request.url)
        for category in response.css('div.bookShelfInfo'):
            yield {
                'ISBN': parse_qs(url.query)['isbn'][0],
                'category': category.css('p a::text').extract_first()
            }
